"""GDScript (Godot Engine) extractor.

Parses ``.gd`` files with a tree-sitter GDScript grammar and emits Graphify's
node/edge dicts. Captures:

* ``class_name X`` / inner ``class X`` -> class node
* ``extends Base`` / ``extends "res://x.gd"`` -> ``extends`` edge
* ``func f(): ...`` -> function node + ``defines`` edge
* generic ``foo()`` / ``obj.method()`` calls inside a body -> ``calls`` edges
* ``Autoload.method()`` -> ``calls`` edge resolved to the function in the
  autoload's script (autoload map read from ``project.godot``)
* ``signal s(args)`` -> signal node + ``declares`` edge
* ``emit_signal("s")`` / ``s.emit()`` -> ``emits`` edge
* ``s.connect(handler)`` -> ``connects`` edge
* ``preload("res://y.gd")`` / ``load("res://y.gd")`` -> ``imports`` edge

Depends on the GDScript grammar from ``tree-sitter-language-pack`` (the Godot
extra, ``graphify[godot]``). If the grammar is not installed the extractor
degrades to a bare file node so the pipeline never crashes.
"""
from __future__ import annotations

import re

from os.path import normpath
from pathlib import Path

from graphify.extractors.base import _file_stem, _make_id

# Cache of resolved [autoload] maps keyed by project root, so project.godot is
# parsed once per worker process rather than per .gd file.
_AUTOLOAD_CACHE: dict[str, dict[str, Path]] = {}
_AUTOLOAD_SECTION_RE = re.compile(r'^\s*\[(?P<name>\w+)\]')

# Engine types are not project definitions. Without this filter every script
# that does ``extends Node3D`` hangs an edge on one ubiquitous ``Node3D`` node,
# which then reads as a core abstraction in god-node/community analysis (#726).
_GDSCRIPT_BUILTIN_TYPES: frozenset[str] = frozenset({
    "Object", "RefCounted", "Reference", "Resource", "Node", "Node2D", "Node3D",
    "Control", "CanvasItem", "Spatial", "Sprite", "Sprite2D", "Sprite3D",
    "Area2D", "Area3D", "RigidBody2D", "RigidBody3D", "CharacterBody2D",
    "CharacterBody3D", "StaticBody2D", "StaticBody3D", "CollisionShape2D",
    "CollisionShape3D", "CollisionObject3D", "Camera2D", "Camera3D", "Button",
    "Label", "Panel", "PanelContainer", "MarginContainer", "VBoxContainer",
    "HBoxContainer", "GridContainer", "TabContainer", "ScrollContainer",
    "LineEdit", "TextEdit", "RichTextLabel", "Tree", "ItemList", "OptionButton",
    "CheckBox", "CheckButton", "Slider", "HSlider", "VSlider", "SpinBox",
    "ProgressBar", "TextureRect", "ColorRect", "AcceptDialog", "ConfirmationDialog",
    "PopupMenu", "MeshInstance3D", "MultiMeshInstance3D", "AudioStreamPlayer",
    "AudioStreamPlayer2D", "AudioStreamPlayer3D", "EditorPlugin", "SubViewport",
    "Timer", "Tween", "AnimationPlayer", "SceneTree", "Viewport", "Window",
    "String", "StringName", "NodePath", "Variant", "RID", "Callable", "Signal",
    "Vector2", "Vector2i", "Vector3", "Vector3i", "Vector4", "Vector4i",
    "Color", "Rect2", "Rect2i", "Transform2D", "Transform3D", "Basis",
    "Quaternion", "Plane", "AABB", "Projection", "Array", "Dictionary",
    "PackedByteArray", "PackedInt32Array", "PackedInt64Array",
    "PackedFloat32Array", "PackedFloat64Array", "PackedStringArray",
    "PackedVector2Array", "PackedVector3Array", "PackedColorArray",
    "Thread", "Mutex", "Semaphore", "Image", "ImageTexture", "Texture2D",
    "Mesh", "ArrayMesh", "SurfaceTool", "Material", "StandardMaterial3D",
    "Shape3D", "BoxShape3D", "SphereShape3D", "CapsuleShape3D",
    "RegEx", "RandomNumberGenerator", "ConfigFile", "StreamPeer",
    "StreamPeerBuffer", "FileAccess", "DirAccess", "JSON", "Time", "OS",
    "Engine", "Input", "InputEvent", "InputEventKey", "InputEventMouseButton",
    "ENetMultiplayerPeer", "MultiplayerAPI", "SceneMultiplayer",
})

# Global functions and constructors: skipped for *bare* ``foo()`` calls. A bare
# call is the only place these are unambiguous - ``size()`` bare is a project
# function, ``arr.size()`` is the engine, so the two skip sets stay separate.
_GDSCRIPT_GLOBAL_BUILTINS: frozenset[str] = frozenset({
    "preload", "load", "print", "printerr", "print_debug", "print_rich",
    "push_error", "push_warning", "assert", "range", "len", "str", "int",
    "float", "bool", "abs", "absf", "absi", "min", "max", "clamp", "clampf",
    "clampi", "round", "roundi", "floor", "floori", "ceil", "ceili", "sign",
    "signf", "signi", "sqrt", "pow", "log", "exp", "sin", "cos", "tan", "asin",
    "acos", "atan", "atan2", "fmod", "fposmod", "posmod", "wrapf", "wrapi",
    "lerp", "lerpf", "lerp_angle", "inverse_lerp", "move_toward", "remap",
    "snapped", "is_equal_approx", "is_zero_approx", "is_instance_valid",
    "is_nan", "is_inf", "randi", "randf", "randf_range", "randi_range",
    "randomize", "seed", "deg_to_rad", "rad_to_deg", "linear_to_db",
    "db_to_linear", "maxf", "maxi", "minf", "mini", "typeof", "super",
    "weakref", "instance_from_id", "hash", "char", "ord", "var_to_str",
    "str_to_var", "bytes_to_var", "var_to_bytes", "angle_difference",
}) | _GDSCRIPT_BUILTIN_TYPES

# Object/Node/Array/Dictionary methods: skipped for ``receiver.method()`` calls.
# These fabricate a phantom target on every receiver they are called through -
# including autoloads, where ``Event.some_signal.emit()`` was resolving to a
# nonexistent ``emit()`` function inside event_bus.gd.
_GDSCRIPT_ENGINE_METHODS: frozenset[str] = frozenset({
    "emit", "connect", "disconnect", "is_connected", "emit_signal", "call",
    "call_deferred", "set_deferred", "callv", "bind", "unbind", "has_method",
    "has_signal", "get_signal_list", "add_child", "remove_child", "get_child",
    "get_children", "get_parent", "get_node", "get_node_or_null", "has_node",
    "find_child", "find_children", "add_to_group", "remove_from_group",
    "is_in_group", "queue_free", "free", "duplicate", "instantiate", "instance",
    "new", "get_tree", "get_viewport", "get_window", "set_process",
    "set_physics_process", "set_process_input", "is_instance_valid",
    "get_instance_id", "add_theme_color_override", "add_theme_font_size_override",
    "add_theme_constant_override", "add_theme_stylebox_override",
    "append", "append_array", "push_back", "pop_back", "pop_front", "insert",
    "remove_at", "erase", "clear", "size", "has", "keys", "values", "find",
    "rfind", "count", "resize", "sort", "sort_custom", "reverse", "slice",
    "is_empty", "front", "back", "fill", "map", "filter", "reduce", "get",
    "set", "merge", "begins_with", "ends_with", "split", "join", "strip_edges",
    "to_lower", "to_upper", "to_int", "to_float", "contains", "substr",
    "format", "replace", "pad_zeros", "left", "right", "length",
    "normalized", "length_squared", "distance_to", "direction_to", "dot",
    "cross", "rotated", "limit_length", "move_toward", "lerp", "slerp",
    "start", "stop", "play", "seek", "resume", "show", "hide", "popup",
    "popup_centered", "grab_focus", "release_focus", "rpc", "rpc_id",
    "is_server", "get_unique_id", "get_remote_sender_id",
    "set_multiplayer_authority", "get_multiplayer_authority",
    "is_multiplayer_authority",
})


def _load_gdscript_parser():
    """Return a tree-sitter Parser for GDScript, or None if no grammar is available.

    Tries grammar sources in order of preference:
      1. ``tree_sitter_language_pack`` — the Godot extra (``graphify[godot]``)
         installs this; it bundles PrestonKnopp's GDScript grammar and is the
         supported path since the standalone wheel is not on PyPI.
      2. ``tree_sitter_gdscript`` — the standalone grammar package, used
         opportunistically if a user has it installed directly.
    """
    try:
        from tree_sitter import Language, Parser
    except Exception:
        return None
    # 1) language-pack (the Godot extra installs this)
    try:
        from tree_sitter_language_pack import get_parser
        return get_parser("gdscript")
    except Exception:
        pass
    # 2) standalone grammar package (opportunistic fallback)
    try:
        import tree_sitter_gdscript as tsg
        return Parser(Language(tsg.language()))
    except Exception:
        pass
    return None


def _txt(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _loc(node) -> str:
    return f"L{node.start_point[0] + 1}"


def _field_ident(node, source: bytes) -> str | None:
    """Return the text of a node's ``name`` field (an identifier)."""
    n = node.child_by_field_name("name")
    if n is not None:
        return _txt(n, source)
    return None


def _first_named(node, types: set[str]):
    for c in node.children:
        if c.is_named and c.type in types:
            return c
    return None


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    return s


def _project_root(path: Path) -> Path | None:
    """Nearest ancestor of ``path`` containing ``project.godot``, else None.

    Returned in the same (usually scan-relative) terms as ``path``, so callers
    can build ids off it without absolutizing - see ``_resolve_res``.
    """
    for parent in [path.parent, *path.parents]:
        try:
            if (parent / "project.godot").exists():
                return parent
        except OSError:
            continue
    return None


def _resolve_res(res_path: str, path: Path) -> Path | None:
    """Resolve a ``res://`` path to a real file relative to the project root.

    The project root is the nearest ancestor containing ``project.godot``;
    fall back to the file's own directory when none is found.

    NEVER absolutize the result. Every id in this module derives from the
    scan-relative path the walker passes in, and the result of this function
    feeds ``_make_id``/``_file_stem`` for *cross-file* targets. An absolute
    path here leaks the on-disk location into those ids, and extract()'s #502
    remap cannot repair it: that pass re-derives ids from the emitting file's
    ``source_file``, which for a cross-file target is the wrong file. The
    result is a permanently dangling edge (#726 follow-up).
    """
    if not res_path.startswith("res://"):
        return None
    rel = res_path[len("res://"):]
    root = _project_root(path) or path.parent
    return Path(normpath(root / rel))


def _autoload_map(path: Path) -> dict[str, Path]:
    """Return ``{AutoloadName: resolved script Path}`` from the project's
    ``project.godot`` ``[autoload]`` section, so that ``Autoload.method()`` calls
    can resolve to the real function node in the autoload's script.

    Only ``.gd`` autoloads are mapped. Result is cached per project root.
    """
    root = _project_root(path)
    if root is None:
        return {}
    key = str(root)
    cached = _AUTOLOAD_CACHE.get(key)
    if cached is not None:
        return cached

    amap: dict[str, Path] = {}
    try:
        text = (root / "project.godot").read_text(encoding="utf-8", errors="replace")
    except OSError:
        _AUTOLOAD_CACHE[key] = amap
        return amap

    section: str | None = None
    for line in text.splitlines():
        m = _AUTOLOAD_SECTION_RE.match(line)
        if m:
            section = m.group("name")
            continue
        if section != "autoload":
            continue
        s = line.strip()
        if not s or "=" not in s:
            continue
        name, _, val = s.partition("=")
        name = name.strip()
        raw = val.strip().strip('"')
        res = raw[1:] if raw.startswith("*") else raw   # '*' = enabled singleton
        resolved = _resolve_res(res, path) if res.endswith(".gd") else None
        if name and resolved is not None:
            amap[name] = resolved
    _AUTOLOAD_CACHE[key] = amap
    return amap


def extract_gdscript(path: Path) -> dict:
    parser = _load_gdscript_parser()
    try:
        raw = path.read_bytes()
    except OSError:
        return {"error": f"cannot read {path}"}

    stem = _file_stem(path)
    file_nid = _make_id(str(path))
    nodes: list[dict] = [{
        "id": file_nid, "label": path.name, "file_type": "code",
        "source_file": str(path), "source_location": None,
    }]
    edges: list[dict] = []
    defined: set[str] = {file_nid}

    def add_node(nid: str, label: str, source_location: str | None = None,
                 source_file: str | None = None) -> None:
        """``source_file`` names the file the node BELONGS to, not the file that
        mentioned it. A stub standing in for another file (a preload/extends
        target) must be attributed to that file: extract()'s post-passes
        namespace node ids by source_file, so a stub attributed to the referrer
        is rewritten into the referrer's namespace and stops matching the node
        the target's own extraction produces - one node per referrer instead of
        one shared node (back_button.tscn had five).
        """
        if nid not in defined:
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": source_file or str(path),
                          "source_location": source_location})
            defined.add(nid)

    def add_edge(src: str, tgt: str, relation: str, location: str | None = None,
                 context: str | None = None, target_file: str | None = None) -> None:
        """``target_file`` is the cross-file resolution hint extract() consumes
        (#2169) to canonicalize an edge endpoint onto the real target node.
        Without it a cross-file target keeps its raw per-file id.
        """
        edge = {"source": src, "target": tgt, "relation": relation,
                "confidence": "EXTRACTED", "confidence_score": 1.0,
                "source_file": str(path), "source_location": location, "weight": 1.0}
        if context:
            edge["context"] = context
        if target_file:
            edge["target_file"] = target_file
        edges.append(edge)

    if parser is None:
        # grammar unavailable: bare file node keeps the pipeline alive
        return {"nodes": nodes, "edges": edges}

    tree = parser.parse(raw)
    root = tree.root_node
    autoloads = _autoload_map(path)   # AutoloadName -> resolved script Path

    # ---- pre-scan: local function names so calls resolve to real defs -------
    local_funcs: dict[str, str] = {}

    def _prescan(node) -> None:
        for c in node.children:
            if c.type in ("function_definition", "constructor_definition"):
                fname = _field_ident(c, raw) or ("_init" if c.type == "constructor_definition" else None)
                if fname:
                    local_funcs.setdefault(fname, _make_id(stem, fname))
            _prescan(c)

    _prescan(root)

    # ---- top-level class identity -------------------------------------------
    owner_nid = file_nid
    owner_label = path.name
    signals: dict[str, str] = {}   # signal name -> node id (script-level)

    cn = _first_named(root, {"class_name_statement"})
    if cn is not None:
        name = _field_ident(cn, raw)
        if name:
            owner_label = name
            owner_nid = _make_id(stem, name)
            add_node(owner_nid, name, _loc(cn))
            add_edge(file_nid, owner_nid, "defines", _loc(cn))

    # ---- extends -------------------------------------------------------------
    ext = _first_named(root, {"extends_statement"})
    if ext is not None:
        type_node = _first_named(ext, {"type"})
        base_txt = _txt(type_node, raw).strip() if type_node is not None else _txt(ext, raw).replace("extends", "", 1).strip()
        if base_txt:
            res = None
            if base_txt.startswith(("\"", "'")):
                res = _resolve_res(_strip_quotes(base_txt), path)
            # NOTE: an ``extends`` to an engine type is deliberately kept. It is
            # one edge per script (not per call site), so it cannot accumulate
            # into a god node the way an engine *method* does, and "which
            # scripts are CharacterBody3D vs Control" is real structure.
            if res is not None:
                tgt = _make_id(str(res))
                add_node(tgt, res.name, source_file=str(res))
                add_edge(owner_nid, tgt, "extends", _loc(ext), target_file=str(res))
            else:
                tgt = _make_id(base_txt)
                add_node(tgt, base_txt)
                add_edge(owner_nid, tgt, "extends", _loc(ext))

    # ---- signals (script level) ---------------------------------------------
    for sig in [c for c in root.children if c.is_named and c.type == "signal_statement"]:
        sname = _field_ident(sig, raw)
        if sname:
            snid = _make_id(stem, "signal:" + sname)
            add_node(snid, sname + " (signal)", _loc(sig))
            add_edge(owner_nid, snid, "declares", _loc(sig))
            signals[sname] = snid

    # ---- functions and their call bodies ------------------------------------
    def handle_call(call_node, func_nid: str) -> None:
        """A bare ``call`` node: callee is the first identifier child."""
        callee = None
        for c in call_node.children:
            if c.type == "identifier":
                callee = _txt(c, raw)
                break
            if c.type in ("attribute",):
                break
        if callee is None:
            return
        args = call_node.child_by_field_name("arguments")
        if callee in ("preload", "load") and args is not None:
            for a in args.children:
                if a.type == "string":
                    res = _resolve_res(_strip_quotes(_txt(a, raw)), path)
                    if res is not None:
                        tgt = _make_id(str(res))
                        add_node(tgt, res.name, source_file=str(res))
                        add_edge(file_nid, tgt, "imports", _loc(call_node),
                                 context="preload", target_file=str(res))
                    break
            return
        if callee == "emit_signal" and args is not None:
            for a in args.children:
                if a.type == "string":
                    sname = _strip_quotes(_txt(a, raw))
                    snid = signals.get(sname) or _make_id(stem, "signal:" + sname)
                    add_node(snid, sname + " (signal)")
                    add_edge(func_nid, snid, "emits", _loc(call_node))
                    break
            return
        # A bare call that resolves to a local function always wins. Otherwise it
        # is either a global builtin or an inherited engine method called with
        # implicit self (``add_child(x)``) - neither is a project call target.
        if callee in local_funcs:
            tgt = local_funcs[callee]
        elif callee in _GDSCRIPT_GLOBAL_BUILTINS or callee in _GDSCRIPT_ENGINE_METHODS:
            return
        else:
            tgt = _make_id(callee)
            add_node(tgt, callee + "()")
        add_edge(func_nid, tgt, "calls", _loc(call_node))

    def handle_attribute_call(attr_node, func_nid: str) -> None:
        """``receiver.method(args)`` -> attribute( identifier, attribute_call )."""
        # Collect EVERY identifier before the call, not just the first: an
        # attribute chain (``PlayerProfile.library.set_active_skin()``) has more
        # than one, and the called method belongs to the LAST link, not the
        # first. Treating the first as the receiver resolved that call into
        # player_profile.gd, which defines no such function - a dangling edge.
        idents: list[str] = []
        acall = None
        for c in attr_node.children:
            if c.type == "identifier":
                idents.append(_txt(c, raw))
            elif c.type == "attribute_call":
                acall = c
        if acall is None:
            return
        recv = idents[0] if idents else None
        is_direct = len(idents) == 1
        method = None
        for c in acall.children:
            if c.type == "identifier":
                method = _txt(c, raw)
                break
        if method is None:
            return
        if method == "connect" and is_direct and recv in signals:
            acargs = acall.child_by_field_name("arguments")
            handler = None
            if acargs is not None:
                for a in acargs.children:
                    if a.type == "identifier":
                        handler = _txt(a, raw)
                        break
            if handler:
                if handler in local_funcs:
                    htgt = local_funcs[handler]
                else:
                    htgt = _make_id(handler)
                    add_node(htgt, handler + "()")
                add_edge(signals[recv], htgt, "connects", _loc(attr_node))
            return
        if method == "emit" and is_direct and recv in signals:
            add_edge(func_nid, signals[recv], "emits", _loc(attr_node))
            return
        # Engine methods exist on every receiver, so resolving them fabricates a
        # target that no script defines. Checked BEFORE autoload resolution:
        # ``Event.some_signal.emit()`` parses as recv=Event/method=emit and was
        # resolving to a nonexistent ``emit()`` inside event_bus.gd.
        if method in _GDSCRIPT_ENGINE_METHODS:
            return
        # ``Autoload.method()`` -> resolve to the real function node in the
        # autoload's script (its id matches _make_id(_file_stem(script), method)).
        if is_direct and recv in autoloads:
            tgt = _make_id(_file_stem(autoloads[recv]), method)
            add_edge(func_nid, tgt, "calls", _loc(attr_node), context=recv,
                     target_file=str(autoloads[recv]))
            return
        tgt = _make_id(method)
        add_node(tgt, method + "()")
        add_edge(func_nid, tgt, "calls", _loc(attr_node), context=(idents[-1] if idents else ""))

    def walk_body(node, func_nid: str) -> None:
        for c in node.children:
            if c.type == "function_definition":
                continue  # nested funcs handled separately
            if c.type == "call":
                handle_call(c, func_nid)
            elif c.type == "attribute":
                # attribute may itself contain an attribute_call (method call)
                if any(k.type == "attribute_call" for k in c.children):
                    handle_attribute_call(c, func_nid)
            walk_body(c, func_nid)

    def collect_functions(container, owner: str) -> None:
        for c in container.children:
            if c.type in ("function_definition", "constructor_definition"):
                fname = _field_ident(c, raw) or ("_init" if c.type == "constructor_definition" else None)
                if not fname:
                    continue
                fnid = _make_id(stem, fname)
                add_node(fnid, fname + "()", _loc(c))
                add_edge(owner, fnid, "defines", _loc(c))
                body = c.child_by_field_name("body")
                if body is not None:
                    walk_body(body, fnid)
            elif c.type == "class_definition":
                iname = _field_ident(c, raw)
                inid = _make_id(stem, iname) if iname else owner
                if iname:
                    add_node(inid, iname, _loc(c))
                    add_edge(owner, inid, "defines", _loc(c))
                cbody = _first_named(c, {"class_body"})
                if cbody is not None:
                    collect_functions(cbody, inid)

    collect_functions(root, owner_nid)

    return {"nodes": nodes, "edges": edges}
