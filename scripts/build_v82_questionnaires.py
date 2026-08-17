#!/usr/bin/env python3
"""Build blinded, standalone v82 questionnaire forms with embedded audio."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path


SCHEMA = "moonmuse-v82-questionnaire-build-v1"
EXPORT_SCHEMA = "moonmuse-v82-listener-export-v1"
ORDER_NAMESPACE = "moonmuse-v82-order-20260809-v1"
FORMS = ("F1", "F2")
FORBIDDEN_PRIVATE_TOKENS = (
    "moonmuse_v70",
    "emo_disentangler",
    "system_private",
    "seed_private",
    "quadrant_private",
    "target_arousal_private",
    "target_valence_private",
    "selection_digest_private",
    "/Users/",
    "/home/",
    "/root/",
)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MoonMuse 匿名音乐试听</title>
  <style>
    :root{--ink:#17233f;--muted:#66708a;--brand:#4c5bb8;--soft:#f3f5fb;--line:#dce1ef;--ok:#16765a;--warn:#a45618;--danger:#b43f35}
    *{box-sizing:border-box} body{margin:0;background:#f7f8fc;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.55}
    main{max-width:860px;margin:0 auto;padding:28px 18px 64px}.card{background:white;border:1px solid var(--line);border-radius:18px;padding:24px;box-shadow:0 8px 28px rgba(28,41,78,.07);margin-bottom:18px}
    h1{font-size:30px;margin:0 0 8px}h2{font-size:21px;margin:0 0 14px}p{margin:8px 0}.muted{color:var(--muted)}.banner{padding:12px 14px;border-radius:12px;background:#fff3e8;color:var(--warn);font-weight:700;margin:14px 0}.launch .banner{background:#eaf7f2;color:var(--ok)}
    label.block{display:block;margin:14px 0 6px;font-weight:700}input[type=text],select,textarea{width:100%;padding:11px 12px;border:1px solid #bfc7db;border-radius:10px;font:inherit;background:white}textarea{min-height:88px;resize:vertical}
    .check{display:flex;align-items:flex-start;gap:9px;margin:12px 0}.check input{margin-top:5px}.consent{background:var(--soft);padding:14px;border-radius:12px;white-space:pre-line}
    button{appearance:none;border:0;border-radius:11px;background:var(--brand);color:white;font:inherit;font-weight:700;padding:11px 17px;cursor:pointer}button.secondary{background:white;color:var(--ink);border:1px solid #aeb7cf}button.danger{background:white;color:var(--danger);border:1px solid #e7b8b3}button:disabled{opacity:.45;cursor:not-allowed}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}
    .progress{height:9px;background:#e7eaf3;border-radius:999px;overflow:hidden}.progress>div{height:100%;background:var(--brand);width:0}.topline{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:8px}.player{margin:16px 0;padding:14px;background:var(--soft);border-radius:14px}audio{width:100%;margin-top:8px}.play-status{font-size:14px;color:var(--muted);margin-top:6px}.play-status.ready{color:var(--ok);font-weight:700}
    fieldset{border:0;padding:0;margin:20px 0}legend{font-weight:800;margin-bottom:10px}.choices{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:9px}.choice{border:1px solid var(--line);border-radius:10px;padding:10px;background:white}.choice label{display:flex;gap:8px;align-items:flex-start;cursor:pointer}.scale{grid-template-columns:repeat(5,1fr)}.scale .choice{text-align:center}.scale label{justify-content:center}.error{color:var(--danger);font-weight:700;min-height:24px}.hidden{display:none!important}.complete{border-left:5px solid var(--ok)}
    @media(max-width:560px){h1{font-size:25px}.card{padding:18px}.scale{grid-template-columns:repeat(5,1fr)}.scale .choice{padding:8px 2px}.topline{align-items:flex-start;flex-direction:column}}
  </style>
</head>
<body class="__BODY_CLASS__">
<main>
  <section id="setup" class="card">
    <h1>MoonMuse 匿名音乐试听</h1>
    <p class="muted">共 24 个短音乐片段，约需 20–30 分钟。请按真实听感作答，不必猜片段来源。</p>
    <div class="banner">__MODE_BANNER__</div>
    <div class="consent">__CONSENT_TEXT__</div>
    <label class="block" for="code">匿名研究编号</label>
    <input id="code" type="text" autocomplete="off" placeholder="由研究者提供">
    <label class="block" for="experience">音乐训练经历（可选）</label>
    <select id="experience"><option value="">不填写</option><option value="none">无系统训练</option><option value="lt3">少于 3 年</option><option value="3to6">3–6 年</option><option value="gt6">6 年以上</option></select>
    <label class="check"><input id="adult" type="checkbox"><span>我已年满 18 周岁。</span></label>
    <label class="check"><input id="consent" type="checkbox"><span>我已阅读上方说明，自愿参加，并知道可以在导出前随时停止。</span></label>
    <label class="check"><input id="headphones" type="checkbox"><span>我会使用耳机或在安静环境中试听，并保持设备音量不变。</span></label>
    <div id="setupError" class="error"></div>
    <div class="actions"><button id="start" type="button">开始试听</button><button id="clear" type="button" class="danger">清除本机记录</button></div>
  </section>

  <section id="survey" class="card hidden">
    <div class="topline"><strong id="position"></strong><span id="formLabel" class="muted"></span></div>
    <div class="progress"><div id="progressBar"></div></div>
    <div class="player"><strong id="clipLabel"></strong><audio id="audio" controls preload="metadata"></audio><div id="playStatus" class="play-status">需连续播放至少 80% 后才能评价。</div></div>
    <div id="questions">
      <fieldset><legend>1. 这段音乐整体更偏向哪种效价？</legend><div class="choices" data-radio="valence"><div class="choice"><label><input type="radio" name="valence" value="positive">正面</label></div><div class="choice"><label><input type="radio" name="valence" value="negative">负面</label></div><div class="choice"><label><input type="radio" name="valence" value="uncertain">不确定</label></div></div></fieldset>
      <fieldset><legend>2. 这段音乐整体更偏向哪种唤醒度？</legend><div class="choices" data-radio="arousal"><div class="choice"><label><input type="radio" name="arousal" value="high">高唤醒（有活力、激动或紧张）</label></div><div class="choice"><label><input type="radio" name="arousal" value="low">低唤醒（平静、舒缓或低沉）</label></div><div class="choice"><label><input type="radio" name="arousal" value="uncertain">不确定</label></div></div></fieldset>
      <fieldset><legend>3. 整体好听 / 可听程度</legend><div class="choices scale" data-radio="quality">__QUALITY_SCALE__</div><div class="muted">1 = 很差，5 = 很好</div></fieldset>
      <fieldset><legend>4. 音乐连贯程度</legend><div class="choices scale" data-radio="coherence">__COHERENCE_SCALE__</div><div class="muted">1 = 很不连贯，5 = 很连贯</div></fieldset>
      <fieldset><legend>5. 你听到了哪些问题？（可多选，也可不选）</legend><div class="choices" id="defects">__DEFECT_CHOICES__</div></fieldset>
      <label class="block" for="notes">补充说明（可选，请勿填写姓名、联系方式等个人信息）</label><textarea id="notes" maxlength="500"></textarea>
    </div>
    <div id="surveyError" class="error"></div>
    <div class="actions"><button id="previous" class="secondary" type="button">上一条</button><button id="next" type="button">保存并下一条</button><button id="home" class="secondary" type="button">返回首页</button></div>
  </section>

  <section id="done" class="card complete hidden">
    <h2>已完成 24 条试听</h2><p>答案只保存在当前浏览器中，不会自动上传。请导出 JSON 并发回研究者；CSV 可作为查看用副本。</p>
    <div class="actions"><button id="exportJson" type="button">导出 JSON</button><button id="exportCsv" class="secondary" type="button">导出 CSV</button><button id="review" class="secondary" type="button">返回检查</button></div>
  </section>
</main>
<script>window.V82_CONFIG=__CONFIG_JSON__;window.V82_AUDIO=__AUDIO_JSON__;</script>
<script>
"use strict";
const CONFIG=window.V82_CONFIG,AUDIO=window.V82_AUDIO,$=id=>document.getElementById(id);
let state=null,index=0,seekingGuard=false;
const requiredRadio=["valence","arousal","quality","coherence"];
function xmur3(str){let h=1779033703^str.length;for(let i=0;i<str.length;i++){h=Math.imul(h^str.charCodeAt(i),3432918353);h=h<<13|h>>>19}return function(){h=Math.imul(h^h>>>16,2246822507);h=Math.imul(h^h>>>13,3266489909);return(h^h>>>16)>>>0}}
function mulberry32(a){return function(){let t=a+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function shuffled(items,material){const seed=xmur3(material)(),rng=mulberry32(seed),out=[...items];for(let i=out.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[out[i],out[j]]=[out[j],out[i]]}return out}
function key(code){return `moonmuse-v82:${CONFIG.study_version}:${CONFIG.form}:${code}`}
function blank(code,experience){return{schema:CONFIG.export_schema,formalStudyData:CONFIG.formalStudyData,study_version:CONFIG.study_version,build_mode:CONFIG.mode,form:CONFIG.form,participant_code:code,musical_experience:experience||"",eligibility:{adult:true,consent:true,headphones_or_quiet:true},started_at:new Date().toISOString(),completed_at:null,order:shuffled(CONFIG.stimulus_ids,`${CONFIG.order_namespace}|${CONFIG.form}|${code}`),responses:{}}}
function load(code,experience){try{const saved=JSON.parse(localStorage.getItem(key(code)));if(saved&&saved.schema===CONFIG.export_schema&&saved.form===CONFIG.form)return saved}catch(_){ }return blank(code,experience)}
function save(){localStorage.setItem(key(state.participant_code),JSON.stringify(state))}
function show(id){for(const name of ["setup","survey","done"])$(name).classList.toggle("hidden",name!==id)}
function validCode(code){return CONFIG.mode==="launch"?/^MM-[A-F0-9]{8}$/.test(code):code.length>=4}
function start(){const code=$("code").value.trim().toUpperCase();if(!validCode(code)){ $("setupError").textContent=CONFIG.mode==="launch"?"请输入研究者提供的有效编号。":"技术测试编号至少 4 个字符。";return}if(!$("adult").checked||!$("consent").checked||!$("headphones").checked){$("setupError").textContent="请确认年龄、同意和试听环境。";return}state=load(code,$("experience").value);index=0;$("setupError").textContent="";render()}
function response(){const id=state.order[index];return state.responses[id]||{stimulus_id:id,played_seconds:0,duration_seconds:0,played_fraction:0,defects:[],notes:"",first_viewed_at:new Date().toISOString()}}
function setRadios(name,value){document.querySelectorAll(`input[name="${name}"]`).forEach(el=>el.checked=el.value===String(value??""))}
function setQuestionEnabled(enabled){$("questions").querySelectorAll("input,textarea").forEach(el=>el.disabled=!enabled)}
function render(){const id=state.order[index],r=response();$("position").textContent=`第 ${index+1} / ${state.order.length} 条`;$("formLabel").textContent=`表单 ${CONFIG.form}`;$("clipLabel").textContent=`匿名片段 ${String(index+1).padStart(2,"0")}`;$("progressBar").style.width=`${(index/state.order.length)*100}%`;$("audio").src=AUDIO[id];$("audio").dataset.furthest=String(r.played_seconds||0);for(const name of requiredRadio)setRadios(name,r[name]);document.querySelectorAll('#defects input[type="checkbox"]').forEach(el=>el.checked=(r.defects||[]).includes(el.value));$("notes").value=r.notes||"";$("previous").disabled=index===0;$("next").textContent=index===state.order.length-1?"保存并完成":"保存并下一条";$("surveyError").textContent="";updatePlayback(r);show("survey")}
function updatePlayback(r){const fraction=Math.max(r.played_fraction||0,Number($("audio").dataset.furthest||0)/($("audio").duration||Infinity));const ready=fraction>=.8;$("playStatus").textContent=ready?`已播放 ${Math.min(100,Math.round(fraction*100))}%，可以评价。`:`已播放 ${Math.round(fraction*100)}%，需达到 80%。`;$("playStatus").classList.toggle("ready",ready);setQuestionEnabled(ready)}
function capturePlayback(){const r=response(),audio=$("audio"),furthest=Math.max(Number(audio.dataset.furthest||0),r.played_seconds||0);r.played_seconds=Number(furthest.toFixed(3));r.duration_seconds=Number((Number.isFinite(audio.duration)?audio.duration:r.duration_seconds||0).toFixed(3));r.played_fraction=r.duration_seconds?Number(Math.min(1,r.played_seconds/r.duration_seconds).toFixed(4)):0;state.responses[r.stimulus_id]=r;return r}
function selected(name){const node=document.querySelector(`input[name="${name}"]:checked`);return node?node.value:""}
function collect(){const r=capturePlayback();if(r.played_fraction<.8){$("surveyError").textContent="请先播放至少 80%。";return false}for(const name of requiredRadio){const value=selected(name);if(!value){$("surveyError").textContent="请完成四个必答评价。";return false}r[name]=name==="quality"||name==="coherence"?Number(value):value}r.defects=[...document.querySelectorAll('#defects input[type="checkbox"]:checked')].map(el=>el.value);r.notes=$("notes").value.trim();r.last_saved_at=new Date().toISOString();state.responses[r.stimulus_id]=r;save();return true}
function next(){if(!collect())return;$("audio").pause();if(index===state.order.length-1){state.completed_at=new Date().toISOString();save();show("done")}else{index++;render()}}
function previous(){capturePlayback();save();$("audio").pause();if(index>0){index--;render()}}
$("audio").addEventListener("seeking",()=>{const audio=$("audio"),furthest=Number(audio.dataset.furthest||0);if(audio.currentTime>furthest+1.0){seekingGuard=true;audio.currentTime=furthest;setTimeout(()=>seekingGuard=false,0)}});
$("audio").addEventListener("timeupdate",()=>{const audio=$("audio");if(seekingGuard||audio.paused||audio.seeking)return;const furthest=Number(audio.dataset.furthest||0);if(audio.currentTime<=furthest+1.5)audio.dataset.furthest=String(Math.max(furthest,audio.currentTime));const r=capturePlayback();updatePlayback(r);save()});
$("audio").addEventListener("ended",()=>{const audio=$("audio");audio.dataset.furthest=String(audio.duration);const r=capturePlayback();updatePlayback(r);save()});
function rows(){return state.order.map((id,pos)=>({...state.responses[id],presentation_position:pos+1}))}
function payload(){return{schema:CONFIG.export_schema,formalStudyData:CONFIG.formalStudyData,study_version:CONFIG.study_version,build_mode:CONFIG.mode,form:CONFIG.form,participant_code:state.participant_code,musical_experience:state.musical_experience,eligibility:state.eligibility,started_at:state.started_at,completed_at:state.completed_at,order:state.order,responses:rows()}}
function download(name,text,type){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function csv(){const header=["schema","formalStudyData","study_version","build_mode","form","participant_code","musical_experience","stimulus_id","presentation_position","valence","arousal","quality","coherence","defects","notes","played_seconds","duration_seconds","played_fraction","first_viewed_at","last_saved_at"];const q=v=>`"${String(v??"").replaceAll('"','""')}"`;const body=rows().map(r=>[CONFIG.export_schema,CONFIG.formalStudyData,CONFIG.study_version,CONFIG.mode,CONFIG.form,state.participant_code,state.musical_experience,r.stimulus_id,r.presentation_position,r.valence,r.arousal,r.quality,r.coherence,(r.defects||[]).join(";"),r.notes,r.played_seconds,r.duration_seconds,r.played_fraction,r.first_viewed_at,r.last_saved_at].map(q).join(","));return [header.join(","),...body].join("\n")}
$("start").onclick=start;$("next").onclick=next;$("previous").onclick=previous;$("home").onclick=()=>{$("audio").pause();show("setup")};$("review").onclick=()=>{index=0;render()};$("exportJson").onclick=()=>download(`moonmuse-v82-${CONFIG.form}-${state.participant_code}.json`,JSON.stringify(payload(),null,2),"application/json");$("exportCsv").onclick=()=>download(`moonmuse-v82-${CONFIG.form}-${state.participant_code}.csv`,`\ufeff${csv()}`,"text/csv;charset=utf-8");$("clear").onclick=()=>{const code=$("code").value.trim().toUpperCase();if(code&&confirm("确认清除这个编号在本机的全部答案？")){localStorage.removeItem(key(code));$("setupError").textContent="已清除。"}};
</script>
</body>
</html>'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def scale_choices(name: str) -> str:
    return "".join(
        f'<div class="choice"><label><input type="radio" name="{name}" value="{value}">{value}</label></div>'
        for value in range(1, 6)
    )


def defect_choices() -> str:
    items = (
        ("repetition", "过度重复"),
        ("overlap_noise", "重叠音或杂音感"),
        ("abrupt_ending", "结尾突然"),
        ("long_pause", "停顿过长"),
        ("overdense_fatiguing", "过密或听着累"),
        ("incoherent", "音符衔接不连贯"),
        ("other", "其他"),
    )
    return "".join(
        f'<div class="choice"><label><input type="checkbox" value="{value}">{label}</label></div>'
        for value, label in items
    )


def load_audio(audio_manifest: Path, audio_root: Path) -> dict[str, dict]:
    rows = read_csv(audio_manifest)
    if len(rows) != 48:
        raise ValueError(f"expected 48 audio rows, found {len(rows)}")
    result = {}
    for row in rows:
        stimulus_id = row["stimulus_id"]
        path = audio_root / Path(row["audio_file"]).name
        if stimulus_id in result:
            raise ValueError(f"duplicate audio ID: {stimulus_id}")
        actual_hash = sha256_file(path)
        if actual_hash != row["audio_sha256"]:
            raise ValueError(f"audio hash mismatch: {path}")
        result[stimulus_id] = {
            "sha256": actual_hash,
            "data": "data:audio/mpeg;base64,"
            + base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    return result


def validate_ethics_record(path: Path | None) -> dict:
    if path is None:
        raise ValueError("launch mode requires --ethics-record")
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "oversight_body",
        "decision",
        "decision_date",
        "investigator_contact",
        "consent_version",
        "data_retention",
        "withdrawal_route",
        "consent_text_zh",
    }
    missing = sorted(key for key in required if not str(record.get(key, "")).strip())
    if missing:
        raise ValueError(f"ethics record is missing fields: {missing}")
    if record["decision"] not in {"approved", "exempt"}:
        raise ValueError("ethics decision must be approved or exempt")
    return record


def build_form(
    form: str,
    stimulus_ids: list[str],
    audio: dict[str, dict],
    mode: str,
    ethics: dict | None,
) -> str:
    if len(stimulus_ids) != 24 or len(set(stimulus_ids)) != 24:
        raise ValueError(f"form {form} must contain 24 unique stimuli")
    missing = sorted(set(stimulus_ids) - set(audio))
    if missing:
        raise ValueError(f"form {form} audio is missing: {missing}")
    formal = mode == "launch"
    consent_text = (
        ethics["consent_text_zh"]
        if ethics
        else "这是预启动技术检查版本，不是正式受试者材料。测试答案不会用于论文分析，也请不要转发给参与者。"
    )
    config = {
        "study_version": "v82-20260809",
        "export_schema": EXPORT_SCHEMA,
        "mode": mode,
        "formalStudyData": formal,
        "form": form,
        "stimulus_ids": sorted(stimulus_ids),
        "order_namespace": ORDER_NAMESPACE,
        "ethics_record_public": (
            {
                key: ethics[key]
                for key in (
                    "oversight_body",
                    "decision",
                    "decision_date",
                    "investigator_contact",
                    "consent_version",
                    "data_retention",
                    "withdrawal_route",
                )
            }
            if ethics
            else None
        ),
    }
    audio_payload = {stimulus_id: audio[stimulus_id]["data"] for stimulus_id in stimulus_ids}
    html = HTML_TEMPLATE
    replacements = {
        "__BODY_CLASS__": "launch" if formal else "prelaunch",
        "__MODE_BANNER__": (
            "正式匿名研究问卷" if formal else "预启动技术检查：禁止发给参与者"
        ),
        "__CONSENT_TEXT__": consent_text,
        "__QUALITY_SCALE__": scale_choices("quality"),
        "__COHERENCE_SCALE__": scale_choices("coherence"),
        "__DEFECT_CHOICES__": defect_choices(),
        "__CONFIG_JSON__": json.dumps(config, ensure_ascii=False, separators=(",", ":")),
        "__AUDIO_JSON__": json.dumps(audio_payload, separators=(",", ":")),
    }
    for marker, value in replacements.items():
        if marker not in html:
            raise AssertionError(f"template marker is missing: {marker}")
        html = html.replace(marker, value)
    lowered = html.lower()
    leaked = [token for token in FORBIDDEN_PRIVATE_TOKENS if token.lower() in lowered]
    if leaked:
        raise ValueError(f"private tokens leaked into {form}: {leaked}")
    return html


def build(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.mode not in {"prelaunch", "launch"}:
        raise ValueError("mode must be prelaunch or launch")
    ethics = validate_ethics_record(args.ethics_record) if args.mode == "launch" else None
    audio = load_audio(args.audio_manifest, args.audio_root)
    args.output_dir.mkdir(parents=True)
    outputs = {}
    for form in FORMS:
        rows = read_csv(args.study_dir / "forms_public" / f"{form}.csv")
        ids = [row["stimulus_id"] for row in rows]
        html = build_form(form, ids, audio, args.mode, ethics)
        path = args.output_dir / f"MoonMuse-v82-{form}-{args.mode}.html"
        path.write_text(html, encoding="utf-8")
        outputs[form] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "stimulus_count": len(ids),
            "audio_sha256": {stimulus_id: audio[stimulus_id]["sha256"] for stimulus_id in ids},
        }
    report = {
        "schema": SCHEMA,
        "status": (
            "launch_questionnaires_ready"
            if args.mode == "launch"
            else "prelaunch_questionnaires_ready_not_for_distribution"
        ),
        "formalStudyData": args.mode == "launch",
        "mode": args.mode,
        "export_schema": EXPORT_SCHEMA,
        "study_manifest_sha256": sha256_file(args.study_dir / "study_manifest.json"),
        "audio_manifest_sha256": sha256_file(args.audio_manifest),
        "ethics_record_sha256": sha256_file(args.ethics_record) if args.ethics_record else None,
        "forms": outputs,
        "private_mapping_embedded": False,
        "external_dependencies": False,
        "remaining_gate": (
            "Researcher technical/browser spot-check only; do not distribute."
            if args.mode == "prelaunch"
            else "Issue only the form matching each private participant code allocation."
        ),
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README-DO-NOT-DISTRIBUTE.txt").write_text(
        (
            "PRELAUNCH TECHNICAL BUILD - DO NOT DISTRIBUTE TO PARTICIPANTS\n"
            if args.mode == "prelaunch"
            else "FORMAL BUILD - DISTRIBUTE ONLY UNDER THE DOCUMENTED ETHICS PROTOCOL\n"
        )
        + "F1 and F2 are separate counterbalanced forms. Participant-code allocation is private.\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--audio-manifest", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("prelaunch", "launch"), default="prelaunch")
    parser.add_argument("--ethics-record", type=Path)
    return parser.parse_args()


def main() -> int:
    report = build(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
