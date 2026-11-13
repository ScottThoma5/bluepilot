class WMACConstants:
  # Lead detection parameters
  LEAD_WINDOW_SIZE = 6  # Stable detection window
  LEAD_PROB = 0.45  # Balanced threshold for lead detection

  # Slow down detection parameters
  SLOW_DOWN_WINDOW_SIZE = 5  # Responsive but stable
  SLOW_DOWN_PROB = 0.3  # Balanced threshold for slow down scenarios
  # BluePilot explorer-st-smooth: urgency above which DEC jumps to blended immediately, skipping the
  # confidence/hysteresis path. Was a hard-coded 0.7 in dec.py. Raised so that moderate (often false)
  # slow-down readings go through the normal ~0.3 s confidence ramp instead of an instant mode flip.
  # MPC FCW still forces blended immediately (unchanged).
  SLOW_DOWN_EMERGENCY_URGENCY = 0.85

  # Optimized slow down distance curve - smooth and progressive
  SLOW_DOWN_BP = [0., 10., 20., 30., 40., 50., 55., 60.]
  SLOW_DOWN_DIST = [32., 46., 64., 86., 108., 130., 145., 165.]

  # Slowness detection parameters
  SLOWNESS_WINDOW_SIZE = 10  # Stable slowness detection
  SLOWNESS_PROB = 0.55  # Clear threshold for slowness
  SLOWNESS_CRUISE_OFFSET = 1.025  # Conservative cruise speed offset
