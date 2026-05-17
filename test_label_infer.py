# -*- coding: utf-8 -*-
"""Quick test for label inference from explanation."""
from src.utils import parse_vlm_output

tests = [
    ("这是一份经过数字篡改的伪造收据。图像中存在多处关键异常：商品单价和总价...", 1),
    ("这是一份经过数字拼接合成的伪造图像，呈现亚特兰大勇士队棒球场内的比赛场景。", 1),
    ("这是一份经过数字篡改的伪造购物收据。收据顶部的手写数字...", 1),
    ("这是一份经过数字篡改的伪造图像，其中COSTAULDALING招牌系后期添加...", 1),
    ("这是一份伪造的英国道路指示牌图像，其中包含多处关键篡改痕迹...", 1),
    ("这是一份真实拍摄的建筑物外墙标牌照片，未发现数字伪造或后期篡改的痕迹。", 0),
    ("这是一份经过数字篡改的伪造图像，其中蓝色敞篷车被人为添加至热带河流场景中...", 1),
    ("这是一份伪造的图像，呈现一台带有品牌标识的自动售货机...", 1),
]

ok_count = 0
for text, expected in tests:
    r = parse_vlm_output(text)
    got = r["label"]
    ok = got == expected
    if ok:
        ok_count += 1
    print(f"{'ok' if ok else 'FAIL'} expected={expected} got={got} | {text[:50]}...")

print(f"\n--- {ok_count}/{len(tests)} passed ---")
