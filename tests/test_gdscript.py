import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from graphify.extract import extract_gdscript, _make_id, _file_stem
from graphify.extractors.gdscript import _load_gdscript_parser

_HAS_GRAMMAR = _load_gdscript_parser() is not None


def _rels(result):
    return {e["relation"] for e in result["edges"]}


def _edge(result, relation):
    return [e for e in result["edges"] if e["relation"] == relation]


@unittest.skipUnless(_HAS_GRAMMAR, "tree-sitter-gdscript grammar not installed")
class TestGDScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "project.godot").write_text("config_version=5\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, body):
        p = self.root / name
        p.write_text(textwrap.dedent(body))
        return p

    def test_class_extends_functions_signals_calls(self):
        (self.root / "audio.gd").write_text("extends Node\nclass_name AudioManager\n")
        p = self._write("enemy.gd", """
            class_name Enemy
            extends CharacterBody2D

            signal died(reason)

            func _ready():
                died.connect(_on_died)
                sprite.play("idle")

            func take_damage(amount):
                if amount > 0:
                    emit_signal("died", "killed")
                    die()

            func die():
                var fx = preload("res://audio.gd")
                queue_free()

            func _on_died(reason):
                print(reason)
        """)
        r = extract_gdscript(p)
        rels = _rels(r)
        for expected in ("defines", "extends", "declares", "emits", "connects", "calls", "imports"):
            self.assertIn(expected, rels, f"missing relation {expected}")

        # class node id
        stem = _file_stem(p)
        enemy_nid = _make_id(stem, "Enemy")
        labels = {n["id"]: n["label"] for n in r["nodes"]}
        self.assertEqual(labels.get(enemy_nid), "Enemy")

        # extends target is the base type
        ext = _edge(r, "extends")[0]
        self.assertEqual(labels.get(ext["target"]), "CharacterBody2D")

        # local call resolves to the DEFINED die() (same id, no orphan anchor)
        die_nid = _make_id(stem, "die")
        calls_targets = {e["target"] for e in _edge(r, "calls")}
        self.assertIn(die_nid, calls_targets)

        # signal connect resolves handler to the local function
        on_died_nid = _make_id(stem, "_on_died")
        conn = _edge(r, "connects")[0]
        self.assertEqual(conn["target"], on_died_nid)

        # preload resolves res:// to the file node id the extractor itself
        # produces for that file. Deriving the expectation with .resolve() would
        # hide the drift this is meant to catch: every id is built from the path
        # the extractor is handed, so an absolutized cross-file target points at
        # a node nobody creates.
        audio_nid = extract_gdscript(self.root / "audio.gd")["nodes"][0]["id"]
        imports_targets = {e["target"] for e in _edge(r, "imports")}
        self.assertIn(audio_nid, imports_targets)

    def test_autoload_method_call_resolves_to_script_function(self):
        # project.godot registers Analytics as an autoload -> analytics.gd
        (self.root / "project.godot").write_text(
            'config_version=5\n\n[autoload]\nAnalytics="*res://analytics.gd"\n'
        )
        analytics = self.root / "analytics.gd"
        analytics.write_text("extends Node\nfunc track(name):\n\tpass\n")
        caller = self._write("player.gd", """
            extends Node
            func attack():
                Analytics.track("hit")
        """)
        r = extract_gdscript(caller)
        # the call must target analytics.gd's track function node, NOT a bare anchor
        want = _make_id(_file_stem(analytics), "track")
        resolved = [e for e in _edge(r, "calls") if e.get("context") == "Analytics"]
        self.assertTrue(resolved, "no autoload-resolved call edge emitted")
        self.assertEqual(resolved[0]["target"], want)
        # a bare `track()` anchor must NOT have been created
        self.assertFalse(any(n["label"] == "track()" for n in r["nodes"]))

    def test_relative_paths_do_not_leak_absolute_location_into_ids(self):
        """Cross-file targets must stay in the scan-relative id namespace.

        The walker hands extractors scan-relative paths. ``res://`` resolution
        used to absolutize, so an autoload/preload target became
        ``f_godot_projects_myproj_autoloads_x_method`` while the node itself was
        created as ``autoloads_x_method`` - a permanently dangling edge. The
        extract() #502 remap cannot repair it: it re-derives ids from the
        *emitting* file's source_file, which for a cross-file target is the
        wrong file.
        """
        (self.root / "project.godot").write_text(
            'config_version=5\n\n[autoload]\nAnalytics="*res://analytics.gd"\n'
        )
        (self.root / "analytics.gd").write_text("extends Node\nfunc track(name):\n\tpass\n")
        (self.root / "caller.gd").write_text(
            "extends Node\nfunc attack():\n\tAnalytics.track('hit')\n\tpreload('res://analytics.gd')\n"
        )
        cwd = Path.cwd()
        os.chdir(self.root)
        try:
            rel_caller = Path("caller.gd")
            r = extract_gdscript(rel_caller)
            # The analytics.gd nodes, as the walker would create them.
            analytics = extract_gdscript(Path("analytics.gd"))
            analytics_ids = {n["id"] for n in analytics["nodes"]}
        finally:
            os.chdir(cwd)

        anchor = _make_id(str(self.root))
        for e in r["edges"]:
            for endpoint in (e["source"], e["target"]):
                self.assertFalse(
                    endpoint.startswith(anchor),
                    f"absolute location leaked into id: {endpoint}",
                )
        # and the cross-file endpoints must actually exist in the target file
        for relation in ("calls", "imports"):
            for e in _edge(r, relation):
                if e.get("context") in ("Analytics", "preload"):
                    self.assertIn(
                        e["target"], analytics_ids,
                        f"{relation} target {e['target']} matches no node in analytics.gd",
                    )

    def test_chained_attribute_is_not_resolved_to_the_autoload(self):
        """``Autoload.member.method()`` belongs to ``member``, not the autoload.

        Taking the first identifier as the receiver resolved
        ``PlayerProfile.library.set_active_skin()`` into player_profile.gd,
        which defines no such function - a permanently dangling edge pointing at
        the wrong file.
        """
        (self.root / "project.godot").write_text(
            'config_version=5\n\n[autoload]\nProfile="*res://profile.gd"\n'
        )
        (self.root / "profile.gd").write_text("extends Node\nvar library\nfunc save():\n\tpass\n")
        p = self._write("menu.gd", """
            extends Node
            func refresh():
                Profile.library.set_active_skin("x")
                Profile.save()
        """)
        r = extract_gdscript(p)
        profile_stem = _file_stem(self.root / "profile.gd")

        # the DIRECT call still resolves into profile.gd
        direct = [e for e in _edge(r, "calls") if e.get("context") == "Profile"]
        self.assertTrue(direct, "direct autoload call was not resolved")
        self.assertEqual(direct[0]["target"], _make_id(profile_stem, "save"))

        # the CHAINED call must NOT be attributed to profile.gd
        self.assertNotIn(
            _make_id(profile_stem, "set_active_skin"),
            {e["target"] for e in _edge(r, "calls")},
            "chained call was wrongly resolved into the autoload's script",
        )

    def test_engine_methods_do_not_fabricate_targets(self):
        """``Autoload.signal.emit()`` must not resolve to a phantom ``emit()``.

        Engine methods exist on every receiver, so resolving them through an
        autoload invents a function node the autoload's script never defines.
        """
        (self.root / "project.godot").write_text(
            'config_version=5\n\n[autoload]\nEvent="*res://event_bus.gd"\n'
        )
        (self.root / "event_bus.gd").write_text("extends Node\nsignal thing_happened\n")
        p = self._write("emitter.gd", """
            extends Node
            func go():
                Event.thing_happened.emit()
                Event.connect("x", y)
                add_child(Node.new())
        """)
        r = extract_gdscript(p)
        labels = {n["label"] for n in r["nodes"]}
        for phantom in ("emit()", "connect()", "add_child()", "new()"):
            self.assertNotIn(phantom, labels, f"engine method leaked as node: {phantom}")

    def test_missing_grammar_is_graceful(self):
        # Even with a grammar present, an empty file yields just the file node.
        p = self._write("empty.gd", "")
        r = extract_gdscript(p)
        self.assertTrue(any(n["label"] == "empty.gd" for n in r["nodes"]))
        self.assertNotIn("error", r)


if __name__ == "__main__":
    unittest.main()
