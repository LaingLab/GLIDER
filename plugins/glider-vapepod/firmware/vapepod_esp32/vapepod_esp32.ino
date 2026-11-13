// VapePod firmware: the vape box side pod's Bluetooth controller.
//
// An ESP32 DevKitC that switches the side pod's atomizer and pump through two
// MOSFET modules, commanded by GLIDER (the glider-vapepod plugin) over
// Bluetooth LE or a USB cable. The protocol and the wiring are in
// ../../README.md.
//
// The pod is built with a XIAO nRF52840 (../vapepod/vapepod.ino). This is the
// same firmware for an ESP32 DevKitC instead; keep the two in step.
//
// The safety rules live here, not in GLIDER, because a host can crash, hang
// or walk out of range:
//   - Both outputs turn off the moment the Bluetooth link drops, and the
//     moment a wired host sends nothing for HOST_TIMEOUT_MS. (The DevKitC's
//     USB-UART bridge can't tell the sketch that a port was closed, so
//     silence is the only sign.) Losing either link turns everything off,
//     so drive the pod over one link at a time.
//   - The atomizer only ever runs for a stated time, at most MAX_PUFF_MS, and
//     then turns itself off. There is no "atomizer on" command, and a longer
//     puff is refused rather than shortened.
//   - If loop() ever stops running, the task watchdog resets the chip, and a
//     reset leaves both outputs low.
//
// Builds on the ESP32 Arduino core 2.x and 3.x (Arduino IDE or PlatformIO).
// Core 4 replaced the Bluetooth API, so it is refused below.

#include <Arduino.h>
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 4
#error "VapePod needs the ESP32 Arduino core 2.x or 3.x (core 4 changed the BLE API). Install esp32 3.3.x in the Boards Manager."
#endif

#define FW_VERSION "0.2.0"

// The advertised name. GLIDER finds the pod by it, so give each pod its own
// name if you build more than one ("VapePod-2", ...).
static const char *DEVICE_NAME = "VapePod";

// GATT layout. glider_vapepod/device.py carries the same three UUIDs.
static const char *SERVICE_UUID = "70c4957d-11e9-41a8-83fc-8783938e1e90";
static const char *COMMAND_UUID = "70c4957d-11e9-41a8-83fc-8783938e1e91";  // write
static const char *STATUS_UUID = "70c4957d-11e9-41a8-83fc-8783938e1e92";   // read, notify

// To the MOSFET modules' signal inputs. Not strapping pins, so the ESP32
// boots the same whatever the modules do to them.
static const uint8_t ATOM_PIN = 25;
static const uint8_t PUMP_PIN = 26;

// The longest puff the pod will run. GLIDER checks against the same figure
// (MAX_PUFF_MS in device.py); change both together.
static const uint32_t MAX_PUFF_MS = 10000;

// 100 % is a solid output with no switching at all. Anything less switches at
// PWM_HZ: slow enough for an optocoupled MOSFET module, but an audible whine
// from the pump.
static const uint32_t PWM_HZ = 1000;
static const uint8_t PWM_BITS = 10;
// Writing all ones is fully on: the core turns (2^bits - 1) into 2^bits.
static const uint32_t PWM_MAX = (1u << PWM_BITS) - 1;

// The status goes out on every change and at least this often, which also
// keeps GLIDER's copy of it fresh enough to record.
static const uint32_t STATUS_EVERY_MS = 1000;

// Wired (USB): one command per line, ended by \n. Once a host has sent a
// command it must keep talking: silence this long counts as a lost link.
// GLIDER sends "status" every second as a heartbeat. Lines the pod sends are
// either a status ("atom=.. pump=..") or a note starting with "# ". Opening
// the port resets a DevKitC (its auto-reset circuit), which leaves both
// outputs low.
static const uint32_t HOST_TIMEOUT_MS = 3000;

static const size_t COMMAND_LEN = 32;

// Types come before the first function: the Arduino builder inserts its
// generated prototypes there, and they need these declared.
struct Output {
  uint8_t pin;
  uint8_t channel;  // LEDC channel; only core 2.x uses it
  uint8_t percent;
};

struct Command {
  char text[COMMAND_LEN];
};

// --- outputs -------------------------------------------------------------------

#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
static void pwmAttach(uint8_t pin, uint8_t) { ledcAttach(pin, PWM_HZ, PWM_BITS); }
static void pwmWrite(uint8_t pin, uint8_t, uint32_t duty) { ledcWrite(pin, duty); }
#else
static void pwmAttach(uint8_t pin, uint8_t channel) {
  ledcSetup(channel, PWM_HZ, PWM_BITS);
  ledcAttachPin(pin, channel);
}
static void pwmWrite(uint8_t, uint8_t channel, uint32_t duty) { ledcWrite(channel, duty); }
#endif

static Output atom = {ATOM_PIN, 0, 0};
static Output pump = {PUMP_PIN, 1, 0};

// When the running puff started and how long it lasts (ms). 0 = no puff.
static uint32_t puffStartedAt = 0;
static uint32_t puffLength = 0;

static BLECharacteristic *statusChar = nullptr;
static uint32_t statusSentAt = 0;

// Set by the Bluetooth callbacks, which run on the Bluetooth task. Commands
// themselves go through `inbox`, so every output change happens in loop()
// and the outputs only ever have one writer.
static volatile bool connected = false;
static volatile bool linkLost = false;
static volatile uint32_t commandsDropped = 0;
static QueueHandle_t inbox;

// The wired link. `wired` is set by the first good command over USB and
// cleared when the host goes quiet; only loop() touches these.
static bool wired = false;
static uint32_t heardFromHostAt = 0;
static char usbLine[COMMAND_LEN];
static size_t usbLength = 0;
static bool usbOverflow = false;

// --- USB serial ----------------------------------------------------------------

// Never blocks: a line that doesn't fit the transmit buffer is dropped rather
// than holding up loop() (and the watchdog).
static void usbSend(const char *text) {
  size_t length = strlen(text);
  if (Serial.availableForWrite() < (int)(length + 1)) {
    return;
  }
  Serial.write((const uint8_t *)text, length);
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

static void setOutput(Output &out, uint8_t percent) {
  out.percent = percent;
  pwmWrite(out.pin, out.channel, (uint32_t)percent * PWM_MAX / 100);
}

static void sendStatus() {
  char text[24];
  snprintf(text, sizeof(text), "atom=%u pump=%u", atom.percent, pump.percent);
  statusChar->setValue((uint8_t *)text, strlen(text));
  if (connected) {
    statusChar->notify();
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

class LinkCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *) override { connected = true; }

  void onDisconnect(BLEServer *) override {
    connected = false;
    linkLost = true;
  }
};

class CommandCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *characteristic) override {
    // std::string on core 2.x, String on 3.x; both have c_str() and length().
    auto value = characteristic->getValue();
    Command command = {};
    size_t length = value.length() < COMMAND_LEN - 1 ? value.length() : COMMAND_LEN - 1;
    memcpy(command.text, value.c_str(), length);
    if (xQueueSend(inbox, &command, 0) != pdTRUE) {
      commandsDropped = commandsDropped + 1;  // reported from loop()
    }
  }
};

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
  pwmAttach(ATOM_PIN, atom.channel);
  pwmAttach(PUMP_PIN, pump.channel);
  allOff();

  Serial.begin(115200);
  note("VapePod %s, max puff %lu ms", FW_VERSION, (unsigned long)MAX_PUFF_MS);
  inbox = xQueueCreate(8, sizeof(Command));

  BLEDevice::init(DEVICE_NAME);
  BLEDevice::setMTU(185);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new LinkCallbacks());
  BLEService *service = server->createService(SERVICE_UUID);

  // Write with response only, so GLIDER learns of a command that never landed.
  BLECharacteristic *command =
      service->createCharacteristic(COMMAND_UUID, BLECharacteristic::PROPERTY_WRITE);
  command->setCallbacks(new CommandCallbacks());

  statusChar = service->createCharacteristic(
      STATUS_UUID, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  statusChar->addDescriptor(new BLE2902());
  service->start();

  // The service UUID goes in the advertisement and the name in the scan
  // response; both together are too long for one 31-byte packet.
  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();
  sendStatus();

  // From here on, a loop() that stops returning resets the chip.
  enableLoopWDT();
}

void loop() {
  if (linkLost) {
    linkLost = false;
    // Whatever was queued was meant for the link that just went.
    xQueueReset(inbox);
    allOff();
    sendStatus();
    note("link lost: all off");
    delay(100);  // let the stack finish the disconnect before advertising again
    BLEDevice::startAdvertising();
  }

  if (commandsDropped != 0) {
    note("inbox full; %lu command(s) dropped", (unsigned long)commandsDropped);
    commandsDropped = 0;
  }

  readUsb();
  if (wired && millis() - heardFromHostAt >= HOST_TIMEOUT_MS) {
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
