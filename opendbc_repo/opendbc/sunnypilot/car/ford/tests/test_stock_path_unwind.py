"""BluePilot: stock curvature-only path (disable_BP_lat_UI on) unwind limits.

Covers STOCK_PATH_UNWIND_LIMITS in values_ext.py as used by carcontroller.apply_ford_curvature_limits:
  * windup is unchanged from the stock table,
  * unwind is faster than stock but stays under ford.h FORD_LIMITS with margin,
  * a turn exit shaped like the 11/16 log (00000324 t~157-166 s) tracks the model instead of trailing it,
  * a Python port of ford.h's steer_angle_cmd_checks (CAN, yaw angle_meas) never flags the output.
"""
import random
import unittest

import numpy as np

from opendbc.car.ford.carcontroller import apply_ford_curvature_limits
from opendbc.car.ford.values import CarControllerParams, FORD_STOCK_ANGLE_LIMITS
from opendbc.sunnypilot.car.ford.values_ext import STOCK_PATH_UNWIND_LIMITS

DT = 0.05  # LateralMotionControl at 20 Hz

# ford.h FORD_LIMITS(false, 100) -- CAN, yaw-sourced angle_meas. Read-only mirror for tests.
PANDA_MAX_ANGLE = 1000
PANDA_DEG_TO_CAN = 50000
PANDA_MAX_ANGLE_ERROR = 100
PANDA_RATE = ([5., 16., 25.], [0.0025, 0.0014, 0.00018])
PANDA_ERROR_MIN_SPEED = 10.0


class _CP:
  flags = 0  # CAN (not CANFD): no lateral-accel clamp in apply_ford_curvature_limits


class PandaFordCurvatureModel:
  """Python port of lateral.h steer_angle_cmd_checks for Ford CAN curvature (wire domain counts).

  Speeds/angle_meas are kept as 6-sample min/max like the firmware's sample buffers.
  """
  def __init__(self):
    self.last = 0
    self.meas = []
    self.speed = []
    self.violations = 0
    self.rate_violations = 0  # violations of the plain rate/max limits (ignoring the measured-curvature window)

  def rx(self, v_ego, wire_meas_curvature):
    self.speed = (self.speed + [v_ego])[-6:]
    self.meas = (self.meas + [int(round(wire_meas_curvature * PANDA_DEG_TO_CAN))])[-6:]

  def tx(self, wire_curvature, enabled=True):
    desired = int(round(wire_curvature * PANDA_DEG_TO_CAN))
    violation = False
    if enabled:
      fudged = min(self.speed) - 1.
      up = int(np.interp(fudged, *PANDA_RATE) * PANDA_DEG_TO_CAN + 1.)
      down = int(np.interp(fudged, *PANDA_RATE) * PANDA_DEG_TO_CAN + 1.)
      hi = self.last + (up if self.last > 0 else down)
      lo = self.last - (down if self.last >= 0 else up)
      self.rate_violations += int(desired > hi or desired < lo or abs(desired) > PANDA_MAX_ANGLE)
      if self.speed[-1] > PANDA_ERROR_MIN_SPEED:
        fe = max(self.speed) + 1.
        up_r = int(np.interp(fe, *PANDA_RATE) * PANDA_DEG_TO_CAN - 1.)
        dn_r = up_r
        lo_e = min(self.meas) - PANDA_MAX_ANGLE_ERROR - 1
        hi_e = max(self.meas) + PANDA_MAX_ANGLE_ERROR + 1
        if self.last > hi_e:
          hi = max(self.last - (dn_r if self.last >= 0 else up_r), hi_e)
        elif self.last < lo_e:
          lo = min(self.last + (dn_r if self.last <= 0 else up_r), lo_e)
        else:
          hi = min(hi, hi_e)
          lo = max(lo, lo_e)
        lo = int(np.clip(lo, -PANDA_MAX_ANGLE, PANDA_MAX_ANGLE))
        hi = int(np.clip(hi, -PANDA_MAX_ANGLE, PANDA_MAX_ANGLE))
      violation = desired > hi or desired < lo
    violation |= abs(desired) > PANDA_MAX_ANGLE
    self.last = desired
    self.violations += int(violation)
    return violation


def step(desired, last, current, v, limits=None):
  return apply_ford_curvature_limits(desired, last, current, v, 0., True, _CP, limits)


class TestStockPathUnwindLimits(unittest.TestCase):
  def test_windup_matches_stock(self):
    self.assertEqual(STOCK_PATH_UNWIND_LIMITS.ANGLE_RATE_LIMIT_UP, FORD_STOCK_ANGLE_LIMITS.ANGLE_RATE_LIMIT_UP)
    self.assertEqual(STOCK_PATH_UNWIND_LIMITS.STEER_ANGLE_MAX, FORD_STOCK_ANGLE_LIMITS.STEER_ANGLE_MAX)
    for v in np.linspace(0, 40, 81):
      for last in (0.0, 0.004, -0.004, 0.015):
        target = last + np.sign(last or 1) * 0.01
        self.assertAlmostEqual(step(target, last, last, v, STOCK_PATH_UNWIND_LIMITS), step(target, last, last, v))

  def test_default_limits_unchanged(self):
    self.assertIs(CarControllerParams.ANGLE_LIMITS, FORD_STOCK_ANGLE_LIMITS)
    self.assertAlmostEqual(step(0.0, 0.02, 0.02, 5.0), 0.02 - 0.00045)

  def test_unwind_faster_than_stock_and_under_panda(self):
    for v in np.linspace(0, 40, 161):
      stock = np.interp(v, *FORD_STOCK_ANGLE_LIMITS.ANGLE_RATE_LIMIT_DOWN)
      new = np.interp(v, *STOCK_PATH_UNWIND_LIMITS.ANGLE_RATE_LIMIT_DOWN)
      panda = np.interp(v - 1., *PANDA_RATE)  # firmware fudges speed down 1 m/s
      self.assertGreaterEqual(new + 1e-12, stock, f"slower than stock at {v}")
      self.assertLessEqual(new, panda * 0.85, f"too close to ford.h at {v}")
      if v <= 16:
        self.assertGreater(new, 2.0 * stock, f"no real unwind gain at {v}")

  def test_logged_right_turn_exit_tracks_model(self):
    # Shape of the 11/16 worst right exit: command pinned at the 0.02 cap, model unwinds
    # 0.028 -> -0.003 over ~2 s while speed rises 7.3 -> 9.5 m/s (OP sign: + = right).
    des = np.concatenate([np.full(10, 0.028), np.linspace(0.028, -0.003, 40), np.full(40, -0.003)])
    v = np.linspace(7.3, 9.5, len(des))
    lag = {}
    for name, lim in (('stock', None), ('new', STOCK_PATH_UNWIND_LIMITS)):
      last = 0.02
      out = []
      for d, vv in zip(des, v, strict=True):
        last = step(d, last, last, vv, lim)  # current = last: no deviation clip below 9 m/s anyway
        out.append(last)
      out = np.array(out)
      err = out - np.clip(des, -0.02, 0.02)
      lag[name] = DT * np.sum(err > 0.002)
      if name == 'new':
        self.assertLessEqual(err.max(), 0.0025)
        self.assertTrue(np.all(out[10:] <= 0.02 + 1e-9))
        # no overshoot past the model on the far side once the model is back under the cap
        unwinding = des < 0.02
        self.assertGreaterEqual((out - des)[unwinding].min(), -1e-6)
    self.assertGreater(lag['stock'], 0.5)
    self.assertLess(lag['new'], 0.15)

  def _monte_carlo(self, limits, seed, noise):
    rng = random.Random(seed)
    rate_v = window_v = 0
    for _ in range(200):
      panda = PandaFordCurvatureModel()
      v = rng.uniform(2, 30)
      last = 0.0
      meas = 0.0
      des = 0.0
      for _ in range(200):
        v = float(np.clip(v + rng.uniform(-0.15, 0.15), 0.5, 35))
        if rng.random() < 0.05:
          des = rng.uniform(-0.12, 0.12)  # intersection-sized steps, and sign flips
        meas += (last - meas) * DT / 0.7 + rng.gauss(0, noise)
        for _ in range(5):  # 100 Hz yaw/speed between 20 Hz commands
          panda.rx(v, -meas)  # wire convention: wire = -OP curvature
        last = step(des, last, meas, v, limits)
        self.assertLessEqual(abs(last), 0.02 + 1e-9)
        panda.tx(-last)
      rate_v += panda.rate_violations
      window_v += panda.violations - panda.rate_violations
    return rate_v, window_v

  def test_panda_model_monte_carlo(self):
    # Clean measurement: the PSCM follows the command; no ford.h violations at all.
    self.assertEqual(self._monte_carlo(STOCK_PATH_UNWIND_LIMITS, 1116, 0.0), (0, 0))
    # Noisy yaw (0.0003 1/m per call random walk): ford.h's "move toward angle_meas at the relaxed rate"
    # rule (above 10 m/s) can flag the STOCK table too when the yaw walks away faster than the stock
    # windup row -- that part is pre-existing and not touched here (windup is unchanged). The new unwind
    # row itself must never exceed the plain ford.h rate/max limits.
    new_rate, _ = self._monte_carlo(STOCK_PATH_UNWIND_LIMITS, 1116, 0.0003)
    self.assertEqual(new_rate, 0)

if __name__ == '__main__':
  unittest.main()
