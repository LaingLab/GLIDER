import {AbsoluteFill, Easing, Img, interpolate, random, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';

// The launch splash: a mouse on a paraglider crosses the card and the GLIDER
// wordmark appears behind it. Rendered to src/glider/assets/splash.gif by
// build.sh; the app shows it at half size (a 2x frame on Retina).

export const BG = '#1e1e1e';
export const FLIGHT_FRAMES = 125; // the mouse's crossing; the rest is the hold

const GLIDER_W = 280;
const GLIDER_H = GLIDER_W * (1526 / 1245); // art aspect
const BODY_X = 0.51; // mouse body, as a fraction of the art box
const BODY_Y = 0.77;

// Wind: faint streaks drifting right-to-left, so the mouse reads as flying into the air.
const STREAKS = Array.from({length: 14}, (_, i) => ({
  y: 60 + random(`y${i}`) * 480,
  len: 90 + random(`l${i}`) * 220,
  speed: 14 + random(`s${i}`) * 12, // px per frame
  start: random(`x${i}`) * 1800,
  curl: (random(`c${i}`) - 0.5) * 16,
  alpha: 0.1 + random(`a${i}`) * 0.18,
}));

const Wind = ({frame, width, height, fade}: {frame: number; width: number; height: number; fade: number}) => (
  <svg width={width} height={height} style={{position: 'absolute', inset: 0, opacity: fade}}>
    <defs>
      <linearGradient id="streak" x1="0" x2="1">
        <stop offset="0" stopColor="#fff" stopOpacity="0" />
        <stop offset="0.5" stopColor="#fff" stopOpacity="1" />
        <stop offset="1" stopColor="#fff" stopOpacity="0" />
      </linearGradient>
    </defs>
    {STREAKS.map((s, i) => {
      const span = width + s.len + 200;
      const x = width + 100 - ((s.start + frame * s.speed) % span);
      const y = s.y + Math.sin((frame + i * 7) / 12) * 4;
      return (
        <path
          key={i}
          d={`M ${x} ${y} q ${s.len / 2} ${s.curl} ${s.len} 0`}
          stroke="url(#streak)"
          strokeWidth={2}
          strokeLinecap="round"
          fill="none"
          opacity={s.alpha}
        />
      );
    })}
  </svg>
);

export const Splash = () => {
  const f = useCurrentFrame();
  const {width, height, durationInFrames} = useVideoConfig();
  const textCy = height / 2;

  const t = interpolate(f, [0, FLIGHT_FRAMES], [0, 1], {extrapolateRight: 'clamp', easing: Easing.inOut(Easing.sin)});
  const bodyX = interpolate(t, [0, 1], [-GLIDER_W, width + GLIDER_W]);
  const bob = Math.sin(f / 9) * 10;
  const tilt = Math.cos(f / 9) * 4 - 3;
  // The GIF plays once and holds its last frame, so the wind settles rather than freezing mid-gust.
  const windFade = interpolate(f, [FLIGHT_FRAMES - 10, durationInFrames - 1], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill style={{backgroundColor: BG, borderRadius: 36, overflow: 'hidden'}}>
      <Wind frame={f} width={width} height={height} fade={windFade} />
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: textCy - 110,
          height: 220,
          lineHeight: '220px',
          textAlign: 'center',
          fontFamily: 'Helvetica Neue, Arial, sans-serif',
          fontWeight: 700,
          fontSize: 180,
          letterSpacing: 28,
          paddingLeft: 28, // balance trailing letter-spacing so it centres
          color: '#f2f2f2',
          clipPath: `inset(0 ${Math.max(0, width - bodyX)}px 0 0)`,
        }}
      >
        GLIDER
      </div>
      <Img
        src={staticFile('glider.png')}
        style={{
          position: 'absolute',
          width: GLIDER_W,
          left: bodyX - GLIDER_W * BODY_X,
          top: textCy - GLIDER_H * BODY_Y + bob,
          transform: `rotate(${tilt}deg)`,
          // A halo in the card colour keeps the white mouse legible over the white letters.
          filter: `drop-shadow(0 0 2px ${BG}) drop-shadow(0 0 3px ${BG}) drop-shadow(0 0 4px ${BG})`,
          transformOrigin: `${BODY_X * 100}% ${BODY_Y * 100}%`,
        }}
      />
    </AbsoluteFill>
  );
};
