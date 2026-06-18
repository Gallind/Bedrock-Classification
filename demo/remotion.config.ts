import { Config } from "@remotion/cli/config";

// 1080p H.264, overwrite previous renders. JPEG frames keep the (many) image
// scenes light; GIF scenes are handled deterministically by @remotion/gif.
Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);
Config.setConcurrency(4);
