// VapePod firmware: the vape box side pod's Bluetooth controller.
//
// A Seeed Studio XIAO nRF52840 that switches the side pod's atomizer and pump
// through two MOSFET modules, commanded by GLIDER (the glider-vapepod plugin)
// over Bluetooth LE or a USB cable. The protocol and the wiring are in
// ../../README.md.
// ../vapepod_esp32/ is the same firmware for an ESP32 DevKitC; keep the two in
// step.
//
// The safety rules live here, not in GLIDER, because a host can crash, hang
// or walk out of range:
//   - Both outputs turn off the moment the Bluetooth link drops, and the
//     moment a wired host goes quiet (closes the port, or sends nothing for
//     HOST_TIMEOUT_MS). Losing either link turns everything off, so drive the
//     pod over one link at a time.
//   - The atomizer only ever runs for a stated time, at most MAX_PUFF_MS, and
//     then turns itself off. There is no "atomizer on" command, and a longer
//     puff is refused rather than shortened.
//   - If loop() ever stops running, the watchdog resets the chip, and a reset
//     leaves both outputs low.
//
// Builds with Seeed's "Seeed nRF52 Boards" core, board "Seeed XIAO nRF52840".
// The "mbed-enabled" core has no Bluefruit library, so it is refused below.

#if defined(ARDUINO_ARCH_MBED)
#error "VapePod needs the 'Seeed nRF52 Boards' core (board Seeed XIAO nRF52840), not 'Seeed nRF52 mbed-enabled Boards'."
#endif

#include <Arduino.h>
#include <bluefruit.h>

#define FW_VERSION "0.3.1"

// The advertised name. GLIDER finds the pod by it, so give each pod its own
// name if you build more than one ("VapePod-2", ...).
static const char *DEVICE_NAME = "VapePod";

// GATT layout. glider_vapepod/device.py carries the same three UUIDs.
static const char *SERVICE_UUID = "70c4957d-11e9-41a8-83fc-8783938e1e90";
static const char *COMMAND_UUID = "70c4957d-11e9-41a8-83fc-8783938e1e91";  // write
static const char *STATUS_UUID = "70c4957d-11e9-41a8-83fc-8783938e1e92";   // read, notify

// To the MOSFET modules' signal inputs.
static const uint8_t ATOM_PIN = D6;
static const uint8_t PUMP_PIN = D5;

// The longest puff the pod will run. GLIDER checks against the same figure
// (MAX_PUFF_MS in device.py); change both together.
static const uint32_t MAX_PUFF_MS = 10000;

// 100 % is a solid output with no switching at all. Anything less switches at
// PWM_HZ: slow enough for an optocoupled MOSFET module, but an audible whine
// from the pump. The PWM counter runs at 16 MHz / 16 = 1 MHz, so PWM_MAX
// counts make one period.
static const uint32_t PWM_HZ = 1000;
static const uint16_t PWM_MAX = 1000000 / PWM_HZ;
static_assert(PWM_MAX <= 0x7FFF, "the nRF52 PWM counter is 15 bits");
static HardwarePWM &pwm = HwPWM1;
static const uint32_t PWM_OWNER = 0x564D5044;  // "VPOD": keeps analogWrite() off this PWM

// The status goes out on every change and at least this often, which also
// keeps GLIDER's copy of it fresh enough to record.
static const uint32_t STATUS_EVERY_MS = 1000;

// Wired (USB): one command per line, ended by \n. Once a host has sent a
// command it must keep talking: silence this long counts as a lost link.
// GLIDER sends "status" every second as a heartbeat. Lines the pod sends are
// either a status ("atom=.. pump=..") or a note starting with "# ".
static const uint32_t HOST_TIMEOUT_MS = 3000;

// A loop() that stops for this long resets the chip.
static const uint32_t WATCHDOG_MS = 2000;

static const size_t COMMAND_LEN = 32;
static const size_t STATUS_LEN = 24;

// Types come before the first function: the Arduino builder inserts its
// generated prototypes there, and they need these declared.
struct Output {
  uint8_t pin;
  uint8_t percent;
};

struct Command {
  char text[COMMAND_LEN];
};

static Output atom = {ATOM_PIN, 0};
static Output pump = {PUMP_PIN, 0};

// When the running puff started and how long it lasts (ms). 0 = no puff.
static uint32_t puffStartedAt = 0;
static uint32_t puffLength = 0;

static BLEService podService(SERVICE_UUID);
static BLECharacteristic commandChar(COMMAND_UUID);
static BLECharacteristic statusChar(STATUS_UUID);
static uint32_t statusSentAt = 0;

// Set by the Bluetooth callbacks, which run on Bluefruit's callback task.
// Commands themselves go through `inbox`, so every output change happens in
// loop() and the outputs only ever have one writer.
static volatile bool connected = false;
static volatile bool linkLost = false;
static volatile uint32_t commandsDropped = 0;
static QueueHandle_t inbox;

// The wired link. `wired` is set by the first good command over USB and
// cleared when the host goes away; only loop() touches these.
static bool wired = false;
static uint32_t heardFromHostAt = 0;
static char usbLine[COMMAND_LEN];
static size_t usbLength = 0;
static bool usbOverflow = false;

// --- USB serial ----------------------------------------------------------------

// Never blocks: with no host, or a host that isn't reading, the line is
// dropped rather than holding up loop() (and the watchdog).
static void usbSend(const char *text) {
  if (!Serial) {
    return;
  }
  size_t length = strlen(text);
  if (Serial.availableForWrite() < (int)(length + 1)) {
    return;
  }
  Serial.write(text, length);
  Serial.write('\n');
}

// A note for whoever is watching the port: "# " then the message.
static void note(const char *format, ...) {
  char text[64] = "# ";
  va_list args;
  va_start(args, format);
  vsnprintf(text + 2, sizeof(text) - 2, format, args);
  va_end(args);
  usbSend(text);
}

// --- outputs -------------------------------------------------------------------

static void pwmBegin() {
  if (!pwm.takeOwnership(PWM_OWNER)) {
    note("PWM1 was already taken");
  }
  pwm.setClockDiv(PWM_PRESCALER_PRESCALER_DIV_16);
  pwm.setMaxValue(PWM_MAX);
  pwm.addPin(ATOM_PIN);
  pwm.addPin(PUMP_PIN);
  // addPin() leaves the pins at standard drive, a few mA. High drive gives an
  // optocoupled module's input LED enough current; a module that feeds its
  // MOSFET gate directly doesn't care.
  pinMode(ATOM_PIN, OUTPUT_H0H1);
  pinMode(PUMP_PIN, OUTPUT_H0H1);
}

static void setOutput(Output &out, uint8_t percent) {
  out.percent = percent;
  // A full count never reaches the compare point, so 100 % stays high.
  pwm.writePin(out.pin, (uint32_t)percent * PWM_MAX / 100);
}

static void sendStatus() {
  char text[STATUS_LEN];
  snprintf(text, sizeof(text), "atom=%u pump=%u", atom.percent, pump.percent);
  statusChar.write(text);
  if (connected && statusChar.notifyEnabled()) {
    statusChar.notify(text);
  }
  usbSend(text);
  statusSentAt = millis();
}

static void stopPuff() {
  puffLength = 0;
  setOutput(atom, 0);
}

static void allOff() {
  stopPuff();
  setOutput(pump, 0);
}

// --- watchdog ------------------------------------------------------------------

static void startWatchdog() {
  // Paused while a debugger halts the CPU. Once started it can't be stopped;
  // the bootloader keeps feeding it during an upload.
  NRF_WDT->CONFIG = (WDT_CONFIG_HALT_Pause << WDT_CONFIG_HALT_Pos) |
                    (WDT_CONFIG_SLEEP_Run << WDT_CONFIG_SLEEP_Pos);
  NRF_WDT->CRV = WATCHDOG_MS * 32768 / 1000;
  NRF_WDT->RREN = WDT_RREN_RR0_Msk;
  NRF_WDT->TASKS_START = 1;
}

static void feedWatchdog() { NRF_WDT->RR[0] = WDT_RR_RR_Reload; }

// --- commands ------------------------------------------------------------------

// Whole number in [lo, hi], and nothing else in the field.
static bool parseNumber(const char *field, long lo, long hi, long &out) {
  if (field == nullptr || *field == '\0') {
    return false;
  }
  char *end = nullptr;
  long value = strtol(field, &end, 10);
  if (*end != '\0' || value < lo || value > hi) {
    return false;
  }
  out = value;
  return true;
}

// Run one command. Returns false (and changes nothing) if it is not one.
//   off                      atomizer and pump off
//   pump,<percent>           pump at 0-100 %; stays until changed
//   puff,<ms>[,<percent>]    atomizer for ms (1..MAX_PUFF_MS) at percent
//                            (default 100), then off; puff,0 stops a puff
//   status                   send the status now
static bool runCommand(char *text) {
  for (char *c = text; *c; ++c) {
    *c = tolower((unsigned char)*c);
  }
  char *fields[4] = {nullptr};
  int count = 0;
  for (char *save = nullptr, *field = strtok_r(text, ", \r\n", &save);
       field != nullptr && count < 4; field = strtok_r(nullptr, ", \r\n", &save)) {
    fields[count++] = field;
  }
  if (count == 0) {
    return false;
  }
  const char *name = fields[0];
  long a = 0, b = 100;

  if (strcmp(name, "off") == 0 && count == 1) {
    allOff();
  } else if (strcmp(name, "pump") == 0 && count == 2 && parseNumber(fields[1], 0, 100, a)) {
    setOutput(pump, a);
  } else if (strcmp(name, "puff") == 0 && (count == 2 || count == 3) &&
             parseNumber(fields[1], 0, MAX_PUFF_MS, a) &&
             (count == 2 || parseNumber(fields[2], 1, 100, b))) {
    if (a == 0) {
      stopPuff();
    } else {
      // A new puff replaces one already running, from now.
      setOutput(atom, b);
      puffStartedAt = millis();
      puffLength = a;
    }
  } else if (strcmp(name, "status") == 0 && count == 1) {
    // Sent below, like every other command.
  } else {
    return false;
  }
  return true;
}

// --- Bluetooth -----------------------------------------------------------------

static void onConnect(uint16_t) { connected = true; }

static void onDisconnect(uint16_t, uint8_t) {
  connected = false;
  linkLost = true;
}

static void onCommand(uint16_t, BLECharacteristic *, uint8_t *data, uint16_t len) {
  Command command = {};
  size_t length = len < COMMAND_LEN - 1 ? len : COMMAND_LEN - 1;
  memcpy(command.text, data, length);
  if (xQueueSend(inbox, &command, 0) != pdTRUE) {
    commandsDropped = commandsDropped + 1;  // reported from loop()
  }
}

// --- one command, from either link ----------------------------------------------

// Run it and send the status, or say it was refused. Returns whether it ran.
static bool handle(char *text) {
  char original[COMMAND_LEN];
  memcpy(original, text, COMMAND_LEN);  // runCommand splits it in place
  if (runCommand(text)) {
    sendStatus();
    return true;
  }
  note("refused: %s", original);
  return false;
}

// Gather USB bytes into lines and run each one. A line too long to be a
// command is thrown away whole, never run as its first COMMAND_LEN bytes.
static void readUsb() {
  for (int budget = 128; budget > 0 && Serial.available() > 0; --budget) {
    int c = Serial.read();
    if (c < 0) {
      break;
    }
    if (c == '\n' || c == '\r') {
      if (usbOverflow) {
        note("refused: line too long");
      } else if (usbLength > 0) {
        usbLine[usbLength] = '\0';
        if (handle(usbLine)) {
          if (!wired) {
            note("wired host connected");
          }
          wired = true;
          heardFromHostAt = millis();
        }
      }
      usbLength = 0;
      usbOverflow = false;
    } else if (usbLength < COMMAND_LEN - 1) {
      usbLine[usbLength++] = (char)c;
    } else {
      usbOverflow = true;
    }
  }
}

void setup() {
  // Hold both MOSFET inputs low before anything else gets a chance to run.
  pinMode(ATOM_PIN, OUTPUT);
  digitalWrite(ATOM_PIN, LOW);
  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);

  Serial.begin(115200);  // USB CDC: the baud rate is ignored
  pwmBegin();
  allOff();
  note("VapePod %s, max puff %lu ms", FW_VERSION, (unsigned long)MAX_PUFF_MS);
  inbox = xQueueCreate(8, sizeof(Command));

  Bluefruit.begin();
  Bluefruit.setName(DEVICE_NAME);
  Bluefruit.setTxPower(4);
  Bluefruit.Periph.setConnectCallback(onConnect);
  Bluefruit.Periph.setDisconnectCallback(onDisconnect);

  podService.begin();

  // Write with response only, so GLIDER learns of a command that never landed.
  commandChar.setProperties(CHR_PROPS_WRITE);
  commandChar.setPermission(SECMODE_NO_ACCESS, SECMODE_OPEN);
  commandChar.setMaxLen(COMMAND_LEN - 1);
  commandChar.setWriteCallback(onCommand);
  commandChar.begin();

  statusChar.setProperties(CHR_PROPS_READ | CHR_PROPS_NOTIFY);
  statusChar.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  statusChar.setMaxLen(STATUS_LEN);
  statusChar.begin();

  // The service UUID goes in the advertisement and the name in the scan
  // response; both together are too long for one 31-byte packet. Advertising
  // restarts from loop() once the outputs are off, not by itself, so a new
  // link can't land before the old one has been cleaned up.
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addService(podService);
  Bluefruit.ScanResponse.addName();
  Bluefruit.Advertising.restartOnDisconnect(false);
  Bluefruit.Advertising.setInterval(32, 244);  // 20 ms for the first 30 s, then 152.5 ms
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);  // until connected
  sendStatus();

  // From here on, a loop() that stops returning resets the chip.
  startWatchdog();
}

void loop() {
  feedWatchdog();

  if (linkLost) {
    linkLost = false;
    // Whatever was queued was meant for the link that just went.
    xQueueReset(inbox);
    allOff();
    sendStatus();
    note("link lost: all off");
    Bluefruit.Advertising.start(0);
  }

  if (commandsDropped != 0) {
    note("inbox full; %lu command(s) dropped", (unsigned long)commandsDropped);
    commandsDropped = 0;
  }

  readUsb();
  // !Serial: the host closed the port (or the cable went). Otherwise silence.
  if (wired && (!Serial || millis() - heardFromHostAt >= HOST_TIMEOUT_MS)) {
    wired = false;
    usbLength = 0;
    usbOverflow = false;
    allOff();
    sendStatus();
    note("wired host lost: all off");
  }

  Command command;
  if (xQueueReceive(inbox, &command, pdMS_TO_TICKS(1)) == pdTRUE) {
    handle(command.text);
  }

  if (puffLength != 0 && millis() - puffStartedAt >= puffLength) {
    stopPuff();
    sendStatus();
  }

  if (millis() - statusSentAt >= STATUS_EVERY_MS) {
    sendStatus();
  }
}
