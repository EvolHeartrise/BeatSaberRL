"""Map-parsing gaps found by the BeatLeader prefetch (2026-09-30):
v3 serializers that drop zero-valued fields (an initial-tempo bpmEvent
{"m": 180} with no "b"; an object on beat 0 with no "b"), and charts
whose Info.dat names files with a non-.dat extension (VENTEN's ben.ben),
which the extraction allowlist used to drop."""
import io
import json
import os
import zipfile

from beat_arena_env import beatsaver, scoresaber


def _info_v2(files):
    return {"_version": "2.1.0", "_songName": "t", "_beatsPerMinute": 120,
            "_songTimeOffset": 0,
            "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {"_difficulty": d, "_beatmapFilename": f,
                     "_noteJumpMovementSpeed": 16, "_noteJumpStartBeatOffset": 0}
                    for d, f in files]}]}


def _v3(bpm_events, first_note_beat_omitted=True):
    n0 = {"x": 1, "y": 0, "c": 0, "d": 1}          # no "b": beat 0
    if not first_note_beat_omitted:
        n0["b"] = 0
    return {"version": "3.3.0", "bpmEvents": bpm_events,
            "colorNotes": [n0, {"b": 4, "x": 2, "c": 1, "d": 1},
                           {"b": 8, "x": 1, "y": 1, "d": 0}],
            "bombNotes": [{"b": 6, "x": 3}], "obstacles": [],
            "sliders": [], "burstSliders": []}


def _write(tmp_path, info, charts):
    d = tmp_path / "m"
    d.mkdir()
    (d / "Info.dat").write_text(json.dumps(info))
    for name, body in charts.items():
        (d / name).write_text(json.dumps(body))
    return str(d)


def test_v3_omitted_beat_fields_default_to_zero(tmp_path):
    d = _write(tmp_path, _info_v2([("Expert", "ExpertStandard.dat")]),
               {"ExpertStandard.dat": _v3([{"m": 120}])})
    omitted = beatsaver.load_beatmap(d, "Expert")
    d2 = tmp_path / "m2"
    os.rename(d, d2)
    d3 = _write(tmp_path, _info_v2([("Expert", "ExpertStandard.dat")]),
                {"ExpertStandard.dat": _v3([{"b": 0, "m": 120}],
                                           first_note_beat_omitted=False)})
    explicit = beatsaver.load_beatmap(d3, "Expert")
    assert repr(omitted) == repr(explicit)


def test_named_chart_with_odd_extension_is_extracted(tmp_path):
    info = _info_v2([("Easy", "ben.ben"), ("Hard", "benben.ben")])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Info.dat", json.dumps(info))
        z.writestr("ben.ben", json.dumps(_v3([{"b": 0, "m": 120}])))
        z.writestr("benben.ben", json.dumps(_v3([{"m": 120}])))
        z.writestr("cover.jpg", b"\xff\xd8")
        z.writestr("evil.exe", b"MZ")
        z.writestr("song.egg", b"OggS")
    buf.seek(0)
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(buf) as z:
        scoresaber._extract_chart_files(z, str(dest))
    assert sorted(os.listdir(dest)) == ["Info.dat", "ben.ben", "benben.ben"]
    assert not scoresaber._missing_named_charts(str(dest))
    os.remove(dest / "benben.ben")              # a pre-fix cache
    assert scoresaber._missing_named_charts(str(dest))
    beatsaver.load_beatmap(str(dest), "Easy")   # parses
