import {Composition} from 'remotion';
import {Splash} from './Splash';

export const Root = () => (
  <Composition id="Splash" component={Splash} durationInFrames={145} fps={30} width={1200} height={600} />
);
