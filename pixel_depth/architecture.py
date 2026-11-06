# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-free SVG diagram for the current PixelDepth model."""

from html import escape
from pathlib import Path


def _box(x, y, width, height, title, lines, *, fill="#eef4ff", stroke="#315b9b", dashed=False):
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    text = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="14" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>',
        f'<text x="{x + 16}" y="{y + 29}" class="title">{escape(title)}</text>',
    ]
    for index, line in enumerate(lines):
        text.append(
            f'<text x="{x + 16}" y="{y + 58 + index * 22}" class="body">{escape(line)}</text>'
        )
    return "\n".join(text)


def _arrow(x1, y1, x2, y2, *, dashed=False, label=None):
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    result = [
        f'<path d="M {x1} {y1} L {x2} {y2}" class="arrow"{dash}/>',
    ]
    if label:
        result.append(
            f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 8}" class="label">'
            f'{escape(label)}</text>'
        )
    return "\n".join(result)


def write_architecture_svg(config, output, *, title="PixelDepth — arquitetura simplificada"):
    """Write an SVG overview using the supplied :class:`PixelDepthConfig`.

    The diagram intentionally shows logical blocks rather than every layer or
    tensor operation.  It is generated from the active model configuration so
    layer count, hidden size, patch size and enabled heads remain visible.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rays = "habilitado" if config.predict_rays else "desabilitado"
    cameras = "habilitado" if config.predict_cameras else "desabilitado"
    confidence = "depth + ray confidence" if config.depth_confidence else "sem confidence de depth"
    attention = (
        f"{config.num_layers} blocos; local + global a partir de {config.alt_start}"
        if config.alt_start != -1
        else f"{config.num_layers} blocos; atenção local"
    )
    taps = ", ".join(str(layer) for layer in config.out_layers)

    width, height = 1500, 930
    parts = [
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}">
<title>{escape(title)}</title>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#49627f"/>
  </marker>
  <style>
    .title {{ font: 700 18px sans-serif; fill: #172b4d; }}
    .body {{ font: 14px sans-serif; fill: #29415f; }}
    .label {{ font: 12px sans-serif; fill: #49627f; text-anchor: middle; }}
    .section {{ font: 700 20px sans-serif; fill: #172b4d; }}
    .note {{ font: 13px sans-serif; fill: #49627f; }}
    .arrow {{ fill: none; stroke: #49627f; stroke-width: 2.5; marker-end: url(#arrow); }}
  </style>
</defs>
<rect width="100%" height="100%" fill="#fbfcfe"/>
<text x="40" y="48" class="section">{escape(title)}</text>
<text x="40" y="74" class="note">Gerado a partir da configuração ativa; blocos internos estão resumidos.</text>
''',
        _box(40, 250, 190, 150, "Entrada", [
            "images: [B, S, C, H, W]",
            f"C = {config.in_channels} canais",
            "várias vistas opcionais",
        ], fill="#e9f7ef", stroke="#2f855a"),
        _box(285, 250, 205, 150, "Patch embedding", [
            f"patch = {config.patch_size} × {config.patch_size}",
            f"hidden = {config.hidden_size}",
            "tokens + camera tokens",
        ]),
        _box(545, 170, 340, 310, "Backbone PixelSeg / DA3", [
            attention,
            f"GQA: {config.num_attention_heads} heads / {config.num_key_value_heads} KV",
            f"features exportadas: [{taps}]",
            "blocos locais preservam cada vista",
            "blocos globais misturam as vistas",
            "RMSNorm + RoPE espacial",
        ], fill="#f0edff", stroke="#7654a6"),
        _box(950, 135, 260, 350, "DualDPT", [
            "projeção + fusão multiescala",
            "duas branches independentes",
            "decoder de alta resolução",
            f"ray branch: {rays}",
        ], fill="#fff5e6", stroke="#bd741c"),
        _box(1280, 125, 180, 105, "Depth", [
            "profundidade positiva",
            confidence,
        ], fill="#e8f4ff", stroke="#2471a3"),
        _box(1280, 265, 180, 105, "Ray map", [
            "origem + direção",
            "6 canais por pixel",
        ], fill="#e8f4ff", stroke="#2471a3"),
        _box(1280, 405, 180, 105, "Câmera", [
            f"decoder: {cameras}",
            "extrinsics + intrinsics",
        ], fill="#fcecef", stroke="#a23b52", dashed=not config.predict_cameras),
        _box(285, 590, 205, 125, "Câmeras conhecidas", [
            "extrinsics + intrinsics",
            "pose encoding → token",
            "condicionamento opcional",
        ], fill="#fff8e8", stroke="#b7791f", dashed=True),
        _box(545, 590, 340, 125, "Supervisão no treino", [
            "depth loss + confidence",
            "ray loss + teacher depth",
            "camera loss quando há poses",
        ], fill="#f2f4f7", stroke="#687385", dashed=True),
        _box(950, 590, 260, 125, "Saídas do modelo", [
            "depth / depth_conf",
            "ray / ray_conf",
            "pose_encoding e câmeras",
        ], fill="#e9f7ef", stroke="#2f855a"),
        _arrow(230, 325, 285, 325),
        _arrow(490, 325, 545, 325),
        _arrow(885, 325, 950, 300),
        _arrow(1210, 230, 1280, 178),
        _arrow(1210, 310, 1280, 318),
        _arrow(1210, 400, 1280, 458),
        _arrow(385, 590, 385, 400, dashed=True, label="se fornecidas"),
        _arrow(715, 590, 715, 480, dashed=True, label="targets"),
        _arrow(1080, 485, 1080, 590, dashed=True),
        '<text x="40" y="855" class="note">Linhas tracejadas = caminho opcional de condicionamento/supervisão.</text>',
        '<text x="40" y="878" class="note">Treino full-da3 usa depth + ray + teacher; camera loss depende de poses no dataset.</text>',
        '</svg>',
    ]
    output.write_text("\n".join(parts), encoding="utf-8")
    return output
