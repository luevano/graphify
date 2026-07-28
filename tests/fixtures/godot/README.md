# Godot test fixtures

Vendored from upstream PR #1929 (`feat/gdscript-godot-extractor`, commit
`de0f6b2`), which added them alongside the Godot extractor. Files are copied
verbatim; only this README is local.

`resource/` is a miniature but complete Godot project — `project.godot`
registering an autoload and a main scene, a `.tscn` attaching a script and
instancing another scene, and the scripts they point at. That cross-file shape
is what makes it worth keeping: the duplicate-file-node and stub-eviction bugs
both needed several files referencing each other, and neither is reachable from
a single-file test.

`gdscript/` is the equivalent set for the `.gd` extractor.
