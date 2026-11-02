"""Sensitivity of the replay impact to the least certain anchors / knobs."""
import impact
from srom_model import ANCHORS, ShoulderROM

for er in (-10, -45):
    an = dict(ANCHORS)
    an["flex90"] = (90.0, 90.0, er, 130.0)
    impact.report(ShoulderROM(anchors=an), f"flex90 ER anchor {er} (default -25)")
an = dict(ANCHORS)
an["side"] = (0.0, 0.0, -90.0, 77.0)
an["abd90"] = (90.0, 0.0, -110.0, 60.0)
impact.report(ShoulderROM(anchors=an), "ER +10 deg at side / 90 abd")
impact.report(ShoulderROM(s=0.0), "zone start 0 (always centring)")
