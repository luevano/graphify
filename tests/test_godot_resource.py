import tempfile
import textwrap
import unittest
from pathlib import Path

from graphify.extract import (
    extract, extract_godot_resource, _make_id, _file_stem, _file_node_id,
)
from graphify.extractors.gdscript import extract_gdscript
from graphify.extractors import godot_resource as gs


def _edges(result, relation):
    return [e for e in result["edges"] if e["relation"] == relation]


def _norm(result):
    """Order-independent snapshot of a result's nodes and edges."""
    n = sorted((x["id"], x["label"], x.get("source_location") or "")
               for x in result["nodes"])
    e = sorted((x["source"], x["target"], x["relation"],
                x.get("context") or "", x.get("source_location") or "")
               for x in result["edges"])
    return n, e


class TestGodotFileIdentity(unittest.TestCase):
    """One file must be ONE node, however many other files reference it.

    Cross-file stubs used to be attributed to the referrer, and extract()'s
    post-passes namespace node ids by source_file - so a script referenced from
    project.godot and from three scenes became four disconnected nodes instead
    of one hub. These assertions run through extract(), not the raw extractor,
    because the split only happens in those post-passes.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _ids_for(self, result, basename):
        return {n["id"] for n in result["nodes"] if n.get("label") == basename}

    def test_script_referenced_from_project_and_scenes_is_one_node(self):
        (self.root / "scripts").mkdir()
        (self.root / "scenes").mkdir()
        (self.root / "project.godot").write_text(
            'config_version=5\n\n[application]\nrun/main_scene="res://scenes/main.tscn"\n'
            '\n[autoload]\nShared="*res://scripts/shared.gd"\n'
        )
        (self.root / "scripts" / "shared.gd").write_text(
            "extends Node\nfunc ping():\n\tpass\n"
        )
        for scene in ("main.tscn", "other.tscn"):
            (self.root / "scenes" / scene).write_text(
                '[gd_scene load_steps=2 format=3]\n\n'
                '[ext_resource type="Script" path="res://scripts/shared.gd" id="1_s"]\n\n'
                '[node name="Root" type="Node"]\n'
                'script = ExtResource("1_s")\n'
            )
        files = [self.root / "project.godot", self.root / "scripts" / "shared.gd",
                 self.root / "scenes" / "main.tscn", self.root / "scenes" / "other.tscn"]
        r = extract(files, cache_root=self.root)

        ids = self._ids_for(r, "shared.gd")
        self.assertEqual(len(ids), 1, f"shared.gd split into {len(ids)} nodes: {sorted(ids)}")

        # every cross-file edge must land on that one node
        node_ids = {n["id"] for n in r["nodes"]}
        for rel in ("attaches_script", "script", "instances", "main_scene"):
            for e in _edges(r, rel):
                self.assertIn(e["target"], node_ids,
                              f"{rel} edge target {e['target']} matches no node")

    def test_file_node_id_is_the_canonical_stem(self):
        """The extractor's file node must carry core's id, not one of its own.

        Encoding the extension (``_make_id(str(path))``) minted
        ``scripts_shared_gd`` alongside the canonical ``scripts_shared`` that
        ``_file_node_id`` - and every semantic extractor following the node-ID
        spec - produces, so a doc->code edge and the AST landed on different
        nodes for the same file. An absolute input must still remap to the
        repo-relative form (#502); the extension-less id is the file node's own
        id rather than a symbol prefix, so the remap has to match it exactly.
        """
        (self.root / "scripts").mkdir()
        (self.root / "project.godot").write_text(
            'config_version=5\n\n[autoload]\nShared="*res://scripts/shared.gd"\n'
        )
        (self.root / "scripts" / "shared.gd").write_text(
            "extends Node\nfunc ping():\n\tpass\n"
        )
        script = self.root / "scripts" / "shared.gd"
        # The extractor's own contract, asserted directly: whatever path form it
        # is handed, the file node id is that path's canonical stem. Asserting
        # this through extract() cannot fail, because the remap registers the
        # extension-ful ABSOLUTE form explicitly and repairs it either way.
        self.assertEqual(extract_gdscript(script)["nodes"][0]["id"],
                         _file_node_id(script))
        self.assertEqual(
            extract_godot_resource(self.root / "project.godot")["nodes"][0]["id"],
            _file_node_id(self.root / "project.godot"))

        files = [self.root / "project.godot", script]
        r = extract(files, cache_root=self.root)
        self.assertEqual(self._ids_for(r, "shared.gd"),
                         {_file_node_id(Path("scripts") / "shared.gd")})
        anchor = _make_id(str(self.root))
        leaked = [n["id"] for n in r["nodes"] if anchor in n["id"]]
        self.assertEqual(leaked, [], f"on-disk path leaked into ids: {leaked[:3]}")

    def test_same_stem_scene_and_script_do_not_dangle(self):
        """`back_button.tscn` + `back_button.gd` is the Godot norm, not an edge case.

        Both collapse to one canonical id (the stem drops the extension), so the
        collision pass salts them apart by path. A cross-file `instances` edge
        from a THIRD scene carries neither salt, so it dangled on the now-dead
        un-salted id - the same failure #1475 fixed for foo.h/foo.c.
        """
        (self.root / "ui").mkdir()
        (self.root / "project.godot").write_text(
            'config_version=5\n\n[application]\nrun/main_scene="res://ui/widget.tscn"\n'
        )
        (self.root / "ui" / "widget.gd").write_text("extends Node\nfunc ping():\n\tpass\n")
        (self.root / "ui" / "widget.tscn").write_text(
            '[gd_scene load_steps=2 format=3]\n\n'
            '[ext_resource type="Script" path="res://ui/widget.gd" id="1_w"]\n\n'
            '[node name="Widget" type="Node"]\n'
            'script = ExtResource("1_w")\n'
        )
        (self.root / "ui" / "host.tscn").write_text(
            '[gd_scene load_steps=2 format=3]\n\n'
            '[ext_resource type="PackedScene" path="res://ui/widget.tscn" id="1_x"]\n\n'
            '[node name="Host" type="Node"]\n'
        )
        files = [self.root / "project.godot", self.root / "ui" / "widget.gd",
                 self.root / "ui" / "widget.tscn", self.root / "ui" / "host.tscn"]
        r = extract(files, cache_root=self.root)

        node_ids = {n["id"] for n in r["nodes"]}
        dangling = sorted({
            (e["relation"], endpoint)
            for e in r["edges"]
            for endpoint in (e["source"], e["target"])
            if endpoint not in node_ids
        })
        self.assertEqual(dangling, [], f"dangling edges after collision salt: {dangling}")

        # the scene and the script must still be DISTINCT nodes
        self.assertEqual(len(self._ids_for(r, "widget.gd")), 1)
        self.assertEqual(len(self._ids_for(r, "widget.tscn")), 1)
        self.assertNotEqual(self._ids_for(r, "widget.gd"), self._ids_for(r, "widget.tscn"))

    def test_cross_file_stub_is_attributed_to_the_target_file(self):
        (self.root / "project.godot").write_text("config_version=5\n")
        (self.root / "child.gd").write_text("extends Node\n")
        (self.root / "parent.tscn").write_text(
            '[gd_scene load_steps=2 format=3]\n\n'
            '[ext_resource type="Script" path="res://child.gd" id="1_c"]\n\n'
            '[node name="Root" type="Node"]\n'
            'script = ExtResource("1_c")\n'
        )
        r = extract_godot_resource(self.root / "parent.tscn")
        stub = [n for n in r["nodes"] if n["label"] == "child.gd"]
        self.assertTrue(stub, "no stub node emitted for the referenced script")
        self.assertTrue(
            stub[0]["source_file"].endswith("child.gd"),
            f"stub attributed to the referrer, not the target: {stub[0]['source_file']}",
        )
        # and the edge carries the resolution hint extract() needs
        attach = _edges(r, "attaches_script")
        self.assertTrue(attach[0].get("target_file", "").endswith("child.gd"))


class TestGodotResource(unittest.TestCase):
    """The .tscn/.tres/project.godot extractor (grammar path when available)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        (self.root / "scenes").mkdir()
        (self.root / "project.godot").write_text("config_version=5\n")
        (self.root / "scripts" / "enemy.gd").write_text(
            "class_name Enemy\nextends CharacterBody2D\nfunc take_damage(a):\n\tpass\n"
        )
        (self.root / "scripts" / "game_state.gd").write_text("extends Node\n")
        (self.root / "scenes" / "Bullet.tscn").write_text(
            '[gd_scene format=3 uid="uid://bul"]\n\n[node name="Bullet" type="Area2D"]\n'
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, body):
        p = self.root / name
        p.write_text(textwrap.dedent(body))
        return p

    def test_scene_script_subscene_and_connection(self):
        p = self._write("scenes/Main.tscn", """
            [gd_scene load_steps=3 format=3 uid="uid://abc"]

            [ext_resource type="Script" path="res://scripts/enemy.gd" id="1_e"]
            [ext_resource type="PackedScene" path="res://scenes/Bullet.tscn" id="2_b"]

            [node name="Enemy" type="CharacterBody2D"]
            script = ExtResource("1_e")

            [node name="Hitbox" type="Area2D" parent="."]

            [connection signal="body_entered" from="Hitbox" to="." method="take_damage"]
        """)
        r = extract_godot_resource(p)

        enemy_gd = _make_id(_file_stem(self.root / "scripts" / "enemy.gd"))
        bullet = _make_id(_file_stem(self.root / "scenes" / "Bullet.tscn"))

        attaches = {e["target"] for e in _edges(r, "attaches_script")}
        self.assertIn(enemy_gd, attaches)

        instances = {e["target"] for e in _edges(r, "instances")}
        self.assertIn(bullet, instances)

        # the connection method resolves to the ROOT script's function node id,
        # i.e. the same id the gdscript extractor emits for take_damage()
        stem = _file_stem(self.root / "scripts" / "enemy.gd")
        take_damage_nid = _make_id(stem, "take_damage")
        conn_targets = {e["target"] for e in _edges(r, "connects")}
        self.assertIn(take_damage_nid, conn_targets)

    def test_project_godot_autoloads_and_main_scene(self):
        p = self._write("project.godot", """
            config_version=5

            [application]
            run/main_scene="res://scenes/Main.tscn"

            [autoload]
            GameState="*res://scripts/game_state.gd"
        """)
        r = extract_godot_resource(p)

        self.assertTrue(_edges(r, "autoload"), "no autoload edge emitted")
        self.assertTrue(_edges(r, "main_scene"), "no main_scene edge emitted")

        gstate = _make_id(_file_stem(self.root / "scripts" / "game_state.gd"))
        script_targets = {e["target"] for e in _edges(r, "script")}
        self.assertIn(gstate, script_targets)

    def test_line_parser_fallback_matches(self):
        # Force the dependency-free line parser (as if the grammar were absent)
        # and confirm it still emits the same edges the default path produces.
        scene = self._write("scenes/Main.tscn", """
            [gd_scene load_steps=3 format=3 uid="uid://abc"]

            [ext_resource type="Script" path="res://scripts/enemy.gd" id="1_e"]
            [ext_resource type="PackedScene" path="res://scenes/Bullet.tscn" id="2_b"]

            [node name="Enemy" type="CharacterBody2D"]
            script = ExtResource("1_e")

            [connection signal="body_entered" from="Enemy" to="." method="take_damage"]
        """)
        default = extract_godot_resource(scene)
        saved = gs._RESOURCE_PARSER
        try:
            gs._RESOURCE_PARSER = None          # disable grammar -> line parser
            forced_lines = extract_godot_resource(scene)
        finally:
            gs._RESOURCE_PARSER = saved
        # The line fallback must be at least as capable as the default path here.
        self.assertEqual(_norm(default), _norm(forced_lines))


@unittest.skipUnless(gs._load_resource_parser() is not None,
                     "godot_resource grammar (tree-sitter-language-pack) not installed")
class TestGodotResourceGrammar(unittest.TestCase):
    """Behaviour specific to the grammar front end."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        (self.root / "project.godot").write_text("config_version=5\n")
        (self.root / "scripts" / "enemy.gd").write_text("extends Node\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, body):
        p = self.root / name
        p.write_text(textwrap.dedent(body))
        return p

    def test_grammar_and_line_parsers_agree(self):
        scene = self._write("Main.tscn", """
            [gd_scene load_steps=2 format=3 uid="uid://abc"]

            [ext_resource type="Script" path="res://scripts/enemy.gd" id="1_e"]
            [ext_resource type="Texture2D" path="res://art/hero.png" id="2_t"]

            [node name="Enemy" type="CharacterBody2D"]
            script = ExtResource("1_e")
        """)
        text = scene.read_text()
        via_grammar = gs._build_scene(scene, gs._blocks_from_grammar(text))
        via_lines = gs._build_scene(scene, gs._blocks_from_lines(text))
        self.assertEqual(_norm(via_grammar), _norm(via_lines))

    def test_grammar_survives_bracket_in_quoted_value(self):
        # A ']' inside a quoted attribute value trips the line-regex section
        # matcher but the grammar parses it correctly.
        scene = self._write("Weird.tscn", """
            [gd_scene format=3]

            [ext_resource type="Script" path="res://scripts/enemy.gd" id="1_e"]

            [node name="Odd]Name" type="Node2D"]
            script = ExtResource("1_e")
        """)
        r = extract_godot_resource(scene)
        enemy_gd = _make_id(_file_stem(self.root / "scripts" / "enemy.gd"))
        attaches = {e["target"] for e in _edges(r, "attaches_script")}
        self.assertIn(enemy_gd, attaches)


if __name__ == "__main__":
    unittest.main()


class TestGodotFixtureProject(unittest.TestCase):
    """End-to-end over the vendored mini-project (see fixtures/godot/README.md).

    A whole Godot project in miniature: project.godot registering an autoload and
    a main scene, a .tscn attaching a script and instancing another scene. Both
    Godot id bugs needed exactly this shape - several files naming each other -
    so a single-file test cannot reach them.
    """

    root = Path(__file__).parent / "fixtures" / "godot" / "resource"

    def test_each_file_is_one_canonically_named_node(self):
        files = sorted(p for p in self.root.rglob("*")
                       if p.suffix in (".gd", ".tscn", ".godot"))
        self.assertTrue(files, "fixture project is missing")
        r = extract(files, cache_root=self.root)

        # ids are canonical against the SCAN root, which for a repo-relative
        # fixture path is the repo itself - not the fixture subdirectory.
        repo = Path(__file__).resolve().parents[1]
        for p in files:
            ids = {n["id"] for n in r["nodes"] if n.get("label") == p.name}
            rel = p.resolve().relative_to(repo)
            self.assertEqual(
                ids, {_file_node_id(rel)},
                f"{rel} should be exactly one node named {_file_node_id(rel)}, got {sorted(ids)}")

    def test_every_cross_file_edge_lands_on_a_node(self):
        files = sorted(p for p in self.root.rglob("*")
                       if p.suffix in (".gd", ".tscn", ".godot"))
        r = extract(files, cache_root=self.root)
        node_ids = {n["id"] for n in r["nodes"]}

        seen = 0
        for rel in ("attaches_script", "instances", "main_scene", "script"):
            for e in _edges(r, rel):
                seen += 1
                self.assertIn(e["target"], node_ids,
                              f"{rel} edge target {e['target']} matches no node")
        self.assertGreater(seen, 0, "fixture project produced no cross-file edges")
