"""Compact web dashboard for the Unity RAG Agent."""
import re
import time
import uuid

from flask import Flask, jsonify, render_template_string, request

from core import config
from core.conversations import ConversationManager, get_all_models
from core.projects import get_project_manager
from core.tools import tool_diff_file, tool_list_files, tool_read_file, tool_search_code, tool_write_file
from llm.agent import UnityAgent
from llm.providers import create_provider, provider_for_model
from rag.indexer import build_index, update_index
from rag.retriever import Retriever

app = Flask(__name__)
_agents: dict[str, UnityAgent] = {}
_index_running = False
_drafts: dict[str, list[dict]] = {}
_tasks: dict[str, list[dict]] = {}

DASHBOARD_HTML = """
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unity RAG Agent</title>
<script src="https://cdn.jsdelivr.net/npm/marked@15.0.7/marked.min.js"></script>
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--fg:#c9d1d9;--fg2:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;--border:#30363d;--r:10px}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:13px 'Segoe UI',sans-serif}
button,input,select{font:inherit}.app{display:grid;grid-template-columns:260px 1fr;grid-template-rows:52px 1fr;height:100vh}
.header{grid-column:1/-1;display:flex;gap:8px;align-items:center;padding:0 14px;background:var(--bg2);border-bottom:1px solid var(--border)}
.sidebar{background:var(--bg2);border-right:1px solid var(--border);padding:10px;overflow:auto}.main{display:flex;flex-direction:column;overflow:hidden}
.tabs{display:flex;background:var(--bg2);border-bottom:1px solid var(--border)}.tab{padding:10px 14px;color:var(--fg2);cursor:pointer}.tab.active{color:var(--accent);border-bottom:2px solid var(--accent)}
.tabc{display:none;flex:1;overflow:auto}.tabc.active{display:block}.input,.select,.btn{border:1px solid var(--border);background:var(--bg3);color:var(--fg);border-radius:var(--r);padding:8px 10px}
.btn{cursor:pointer}.btn:hover{border-color:var(--accent)}.btn.primary{background:#238636;border-color:#2ea043;color:#fff}
.row{display:flex;gap:8px;align-items:center}.col{display:flex;flex-direction:column;gap:8px}.grow{flex:1}.muted{color:var(--fg2)}
.toolbar{display:grid;grid-template-columns:1.2fr .8fr .8fr .35fr auto;gap:8px;padding:12px;border-bottom:1px solid var(--border);background:var(--bg2)}
.chat{padding:14px;overflow:auto;height:calc(100vh - 190px)}.msg{margin-bottom:14px;max-width:88%}.msg.user{margin-left:auto}.bubble{border:1px solid var(--border);padding:10px 12px;border-radius:var(--r);background:var(--bg3);line-height:1.6}.user .bubble{background:#1f3a5f;border-color:#2d4a6f}
.conv{padding:8px 10px;border-radius:var(--r);cursor:pointer;border:1px solid transparent}.conv:hover,.conv.active{background:var(--bg3);border-color:var(--border)}.sidebox{border-top:1px solid var(--border);padding-top:10px;margin-top:10px}.tiny{font-size:11px}.task.done{text-decoration:line-through;color:var(--fg2)}
.list,.results,.log{padding:12px}.card{border:1px solid var(--border);border-radius:var(--r);padding:10px;background:var(--bg3);margin-bottom:8px}.mono{font-family:Consolas,monospace;white-space:pre-wrap}.editor{width:100%;min-height:280px;resize:vertical;font-family:Consolas,monospace;white-space:pre;background:#0f141b}.diff{max-height:220px;overflow:auto;background:#0f141b;border-radius:var(--r);padding:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;padding:12px}.stat{border:1px solid var(--border);border-radius:var(--r);padding:12px;background:var(--bg3)}
.value{font-size:22px;font-weight:700}.green{color:var(--green)}.blue{color:var(--accent)}.modal{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center}.modal.show{display:flex}.modalbox{width:min(420px,92vw);background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:18px}
@media(max-width:900px){.app{grid-template-columns:1fr}.sidebar{display:none}.toolbar{grid-template-columns:1fr}}
</style></head><body>
<div class="app">
<div class="header"><b>Unity RAG Agent</b><span id="status" class="muted">未选择工程</span><div class="grow"></div>
<select id="model" class="select" onchange="switchModel(this.value)">{% for m in models %}<option value="{{m.id}}" {{' selected' if m.id==current_model else ''}}>{{m.name}} ({{m.provider}})</option>{% endfor %}</select>
<button class="btn" onclick="showAdd()">+ 工程</button></div>
<div class="sidebar"><div class="col">
<select id="project" class="select" onchange="setProject(this.value)"><option value="">选择工程...</option></select>
<div class="row"><button class="btn primary grow" onclick="newConv()">+ 新对话</button><button class="btn" onclick="refreshAll()">刷新</button></div>
<div class="row"><button class="btn grow" onclick="rebuild()">重建索引</button><button class="btn grow" onclick="updateIdx()">增量更新</button></div>
<div class="muted" style="margin-top:10px">对话</div><div id="convs" class="col"><div class="muted">暂无对话</div></div>
<div class="sidebox"><div class="muted">Suggested Files</div><div id="suggestedFiles" class="col tiny"><div class="muted">暂无建议</div></div></div>
<div class="sidebox"><div class="row"><div class="muted">Draft Queue</div><button class="btn tiny" onclick="applyAllDrafts()">Apply All</button><button class="btn tiny" onclick="discardAllDrafts()">Discard All</button></div><div id="drafts" class="col tiny"><div class="muted">暂无 draft</div></div></div>
<div class="sidebox"><div class="row"><div class="muted">Tasks</div><button class="btn tiny" onclick="addTask()">+ Task</button></div><div id="tasks" class="col tiny"><div class="muted">暂无任务</div></div></div>
</div></div>
<div class="main">
<div class="tabs"><div class="tab active" onclick="tab('chat',this)">对话</div><div class="tab" onclick="tab('search',this)">检索</div><div class="tab" onclick="tab('files',this)">文件</div><div class="tab" onclick="tab('stats',this)">统计</div><div class="tab" onclick="tab('log',this)">日志</div></div>
<div id="chat" class="tabc active"><div class="toolbar"><input id="chatPath" class="input" placeholder="限定路径"><select id="chatDomain" class="select"><option value="">全部域</option><option value="code">代码</option><option value="scene">场景</option><option value="prefab">预制体</option><option value="audio">音频</option><option value="image">图片</option><option value="asset">资产</option></select><input id="chatTopK" class="input" type="number" value="6" min="1"><div class="muted">过滤</div><button class="btn" onclick="clearChat()">清空</button></div><div id="chatMsgs" class="chat"><div class="muted" style="text-align:center;margin-top:40px">选择工程后开始对话</div></div><div class="row" style="padding:12px;border-top:1px solid var(--border);background:var(--bg2)"><input id="chatInput" class="input grow" placeholder="输入问题..." onkeydown="if(event.key==='Enter')sendChat()"><button class="btn primary" onclick="sendChat()">发送</button></div></div>
<div id="search" class="tabc"><div class="toolbar"><input id="searchQ" class="input" placeholder="输入检索查询" onkeydown="if(event.key==='Enter')runSearch()"><select id="searchDomain" class="select"><option value="">全部域</option><option value="code">代码</option><option value="scene">场景</option><option value="prefab">预制体</option><option value="audio">音频</option><option value="image">图片</option><option value="asset">资产</option></select><input id="searchPath" class="input" placeholder="限定路径"><input id="searchTopK" class="input" type="number" value="8" min="1"><button class="btn primary" onclick="runSearch()">检索</button></div><div id="searchR" class="results"><div class="muted">输入查询后显示结果</div></div></div>
<div id="files" class="tabc"><div class="toolbar" style="grid-template-columns:1fr .8fr auto auto auto auto"><input id="lsPath" class="input" placeholder="目录路径" value="."><input id="lsPattern" class="input" placeholder="pattern" value="*"><button class="btn primary" onclick="listFiles()">列目录</button><button class="btn" onclick="readFile()">读取文件</button><button class="btn" onclick="previewDiff()">预览 Diff</button><button class="btn" onclick="saveFile()">保存</button></div><div class="grid" style="grid-template-columns:1fr 1fr"><div class="card"><div class="muted">目录 / 文件</div><div id="filesR" class="list"></div></div><div class="col"><div class="card"><div class="row"><input id="readPath" class="input grow" placeholder="文件路径"><input id="readLines" class="input" type="number" value="200" min="20"><button class="btn" onclick="useEditorInChat()">加入聊天上下文</button></div><textarea id="editPrompt" class="input" style="margin-top:10px;min-height:82px;resize:vertical" placeholder="AI 修改指令，例如：在保持现有风格的前提下，为空状态添加一个提示组件。"></textarea><div class="row" style="margin-top:8px"><button class="btn" onclick="draftEdit()">AI Draft</button><button class="btn" onclick="revertDraft()">Discard Draft</button></div><textarea id="editorR" class="input editor" style="margin-top:10px" placeholder="读取文件后在这里编辑"></textarea><div id="diffR" class="mono diff muted" style="margin-top:10px">Diff 预览会显示在这里</div></div><div class="card"><div class="row"><input id="grepQ" class="input grow" placeholder="搜索文本"><input id="grepPattern" class="input" value="*.cs"></div><div id="grepR" class="list"></div></div></div></div></div>
<div id="stats" class="tabc"><div id="statsR" class="results"><div class="muted">选择工程后查看统计</div></div></div>
<div id="log" class="tabc"><div id="logR" class="log mono">等待操作...</div></div>
</div></div></div>
<div id="addModal" class="modal"><div class="modalbox"><h3>添加工程</h3><div class="col"><input id="addPath" class="input" placeholder="工程路径"><input id="addName" class="input" placeholder="显示名称"><input id="addVersion" class="input" value="2022.3 LTS"><select id="addPipeline" class="select"><option>URP</option><option>HDRP</option><option>Built-in</option></select><div class="row" style="justify-content:flex-end"><button class="btn" onclick="hideAdd()">取消</button><button class="btn primary" onclick="addProject()">添加</button></div></div></div></div>
<script>
let activeProject=null,activeConv=null,currentModel='{{current_model}}',selectedFile='',lastReadContent='',suggestedFiles=[];
const esc=s=>{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML};
const md=t=>{try{return marked.parse(t||'',{breaks:true,gfm:true})}catch(e){return esc(t||'').replace(/\\r?\\n/g,'<br>')}};
const api=async(p,d=null,m=null)=>{const o=d||m?{method:m||'POST',headers:{'Content-Type':'application/json'},body:d?JSON.stringify(d):null}:{},r=await fetch('/api'+p,o),t=await r.text();try{return JSON.parse(t)}catch(e){throw new Error(t||`HTTP ${r.status}`)}};
const log=(m,l='info')=>{const x=document.getElementById('logR');x.innerHTML+=`\\n<div class="${l==='error'?'red':''}">[${new Date().toLocaleTimeString()}] ${esc(m)}</div>`;x.scrollTop=x.scrollHeight};
const tab=(n,e)=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tabc').forEach(x=>x.classList.remove('active'));document.getElementById(n).classList.add('active');e.classList.add('active')};
const showAdd=()=>{document.getElementById('addModal').classList.add('show');document.getElementById('addPath').focus()}; const hideAdd=()=>document.getElementById('addModal').classList.remove('show');
const clearChat=()=>{chatPath.value='';chatDomain.value='';chatTopK.value='6'};
function pushSuggested(paths){for(const path of (paths||[])){if(path&&!suggestedFiles.includes(path))suggestedFiles.unshift(path)} suggestedFiles=suggestedFiles.slice(0,12); renderSuggestedFiles()}
function renderSuggestedFiles(){suggestedFiles=suggestedFiles.filter(Boolean); const html=suggestedFiles.length?suggestedFiles.map(p=>`<div class="card" onclick="selectedFile='${p.replace(/'/g,\"\\\\'\")}'; readPath.value='${p.replace(/'/g,\"\\\\'\")}'; tab('files',document.querySelectorAll('.tab')[2]); readFile();"><span style="color:var(--accent)">${esc(p)}</span></div>`).join(''):'<div class="muted">暂无建议</div>'; document.getElementById('suggestedFiles').innerHTML=html}
async function loadDrafts(){if(!activeProject)return; const d=await api(`/edit/drafts?project_id=${encodeURIComponent(activeProject)}`,null,'GET'); drafts.innerHTML=!d.drafts||!d.drafts.length?'<div class="muted">暂无 draft</div>':d.drafts.map(x=>`<div class="card"><div><span style="color:var(--accent)">${esc(x.path)}</span></div><div class="muted">${esc(x.model||'model')} · ${x.changed?'changed':'no change'}</div><div class="muted">${esc(x.instruction||'')}</div><div class="row"><button class="btn tiny" onclick="openDraft('${x.draft_id}')">Open</button><button class="btn tiny" onclick="applyDraft('${x.draft_id}')">Apply</button><button class="btn tiny" onclick="discardDraft('${x.draft_id}')">Discard</button></div></div>`).join('')}
async function openDraft(id){const d=await api(`/edit/drafts?project_id=${encodeURIComponent(activeProject)}`,null,'GET'); const x=(d.drafts||[]).find(v=>v.draft_id===id); if(!x)return; selectedFile=x.path; readPath.value=x.path; editorR.value=x.proposed_content||''; diffR.textContent=x.diff||'没有变更'; tab('files',document.querySelectorAll('.tab')[2]); pushSuggested([x.path])}
async function applyDraft(id){const d=await api('/edit/apply',{project_id:activeProject,draft_id:id}); if(d.error){log(`Apply 失败: ${d.error}`,'error'); return} log(`已应用 draft: ${d.path}`); await loadDrafts()}
async function applyAllDrafts(){if(!activeProject)return; const d=await api('/edit/apply_all',{project_id:activeProject}); if(d.error){log(`批量 Apply 失败: ${d.error}`,'error'); return} log(`批量 Apply 完成: ${d.applied.length} 个文件`); await loadDrafts()}
async function discardDraft(id){await api('/edit/discard',{project_id:activeProject,draft_id:id}); log('已丢弃 draft'); await loadDrafts()}
async function discardAllDrafts(){if(!activeProject)return; const d=await api('/edit/discard_all',{project_id:activeProject}); log(`已丢弃 ${d.discarded||0} 个 draft`); await loadDrafts()}
async function loadTasks(){if(!activeProject)return; const d=await api(`/tasks?project_id=${encodeURIComponent(activeProject)}`,null,'GET'); tasks.innerHTML=!d.tasks||!d.tasks.length?'<div class="muted">暂无任务</div>':d.tasks.map(t=>`<div class="card ${t.done?'task done':'task'}"><div>${esc(t.text)}</div><div class="row"><button class="btn tiny" onclick="toggleTask('${t.task_id}')">${t.done?'Undo':'Done'}</button><button class="btn tiny" onclick="deleteTask('${t.task_id}')">Delete</button></div></div>`).join('')}
async function addTask(){if(!activeProject)return; const text=prompt('输入任务内容'); if(!text)return; const d=await api('/tasks/add',{project_id:activeProject,text}); if(d.error){log(`任务添加失败: ${d.error}`,'error'); return} await loadTasks()}
async function toggleTask(id){await api('/tasks/toggle',{project_id:activeProject,task_id:id}); await loadTasks()}
async function deleteTask(id){await api('/tasks/delete',{project_id:activeProject,task_id:id}); await loadTasks()}
async function loadProjects(){const d=await api('/projects'),sel=project;sel.innerHTML='<option value=\"\">选择工程...</option>'+d.projects.map(p=>`<option value="${p.project_id}" ${activeProject===p.project_id?'selected':''}>${esc(p.name)}</option>`).join(''); if(!activeProject&&d.active_id){activeProject=d.active_id;sel.value=d.active_id} const a=d.projects.find(p=>p.project_id===activeProject); status.textContent=a?`${a.name} · ${a.path}`:'未选择工程'; if(activeProject){await loadConvs();await loadStats();await loadDrafts();await loadTasks();renderSuggestedFiles()}}
async function setProject(id){if(!id)return;activeProject=id;activeConv=null;suggestedFiles=[];renderSuggestedFiles();await api('/projects/active',{project_id:id});chatMsgs.innerHTML='<div class="muted" style="text-align:center;margin-top:40px">点击“新对话”或直接提问</div>';await refreshAll()}
async function addProject(){const d=await api('/projects/add',{path:addPath.value.trim(),name:addName.value.trim(),unity_version:addVersion.value.trim(),render_pipeline:addPipeline.value}); if(d.error)return alert(d.error); activeProject=d.project_id; activeConv=null; await api('/projects/active',{project_id:d.project_id}); hideAdd(); addPath.value=''; addName.value=''; await refreshAll(); log(`已添加工程 ${d.name}`)}
async function rebuild(){if(!activeProject)return;tab('log',document.querySelectorAll('.tab')[4]);log('开始重建索引');const d=await api('/index/build',{project_id:activeProject}); if(d.error)return log(`重建失败: ${d.error}`,'error'); log(`重建完成: ${d.chunks} chunks / ${d.files} files`); await loadStats()}
async function updateIdx(){if(!activeProject)return;tab('log',document.querySelectorAll('.tab')[4]);log('开始增量更新');const d=await api('/index/update',{project_id:activeProject}); if(d.error)return log(`更新失败: ${d.error}`,'error'); log('更新完成'); await loadStats()}
async function loadConvs(){if(!activeProject)return;const d=await api(`/conversations?project_id=${encodeURIComponent(activeProject)}`,null,'GET'); convs.innerHTML=!d.conversations||!d.conversations.length?'<div class="muted">暂无对话</div>':d.conversations.map(c=>`<div class="conv ${activeConv===c.conv_id?'active':''}" onclick="loadConv('${c.conv_id}')"><div>${esc(c.title)}</div><div class="muted">${esc(c.model)} · ${Math.floor((c.message_count||0)/2)} 轮</div><div class="row"><span class="muted" onclick="event.stopPropagation();renameConv('${c.conv_id}','${(c.title||'').replace(/'/g,\"\\\\'\")}')">重命名</span><span class="muted" onclick="event.stopPropagation();delConv('${c.conv_id}')">删除</span></div></div>`).join('')}
async function newConv(){if(!activeProject)return alert('请先选择工程'); const d=await api('/conversations/new',{project_id:activeProject,model:currentModel}); if(d.error)return alert(d.error); activeConv=d.conv_id; chatMsgs.innerHTML='<div class="muted" style="text-align:center;margin-top:40px">新对话已创建</div>'; await loadConvs()}
async function renameConv(id,title){const next=prompt('输入新标题',title); if(!next)return; const d=await api('/conversations/rename',{project_id:activeProject,conv_id:id,title:next}); if(d.error)return alert(d.error); await loadConvs(); log(`已重命名对话: ${next}`)}
async function loadConv(id){activeConv=id; const d=await api('/conversations/load',{project_id:activeProject,conv_id:id}); if(d.error)return alert(d.error); chatMsgs.innerHTML=''; (d.messages||[]).forEach(m=>{chatMsgs.innerHTML+=m.role==='user'?`<div class="msg user"><div class="bubble">${esc((m.content||'').slice(0,400))}</div></div>`:`<div class="msg"><div class="bubble">${md(m.content)}</div></div>`}); chatMsgs.scrollTop=chatMsgs.scrollHeight; if(d.model){currentModel=d.model; model.value=d.model} await loadConvs()}
async function delConv(id){if(!confirm('确定删除此对话？'))return; await api('/conversations/delete',{project_id:activeProject,conv_id:id}); if(activeConv===id){activeConv=null; chatMsgs.innerHTML='<div class="muted" style="text-align:center;margin-top:40px">对话已删除</div>'} await loadConvs()}
async function sendChat(){if(!activeProject)return alert('请先选择工程'); const msg=chatInput.value.trim(); if(!msg)return; chatInput.value=''; chatMsgs.innerHTML+=`<div class="msg user"><div class="bubble">${esc(msg)}</div></div><div class="msg" id="typing"><div class="bubble muted">思考中...</div></div>`; chatMsgs.scrollTop=chatMsgs.scrollHeight; const d=await api('/chat',{message:msg,project_id:activeProject,conv_id:activeConv,model:currentModel,filter_domain:chatDomain.value||null,filter_path:chatPath.value.trim()||null,top_k:parseInt(chatTopK.value||'6',10)}).catch(e=>({error:e.message})); document.getElementById('typing')?.remove(); if(d.error){chatMsgs.innerHTML+=`<div class="msg"><div class="bubble" style="color:var(--red)">${esc(d.error)}</div></div>`}else{if(d.conv_id){activeConv=d.conv_id; await loadConvs()} pushSuggested((d.sources||[]).map(s=>s.path)); const src=d.sources&&d.sources.length?'<div class="muted" style="margin-top:8px">'+d.sources.map((s,i)=>`[${i+1}] ${esc(s.path)} ${s.collection_domain?`[${esc(s.collection_domain)}]`:''} (${Number(s.score||0).toFixed(2)})`).join('<br>')+'</div>':''; const tools=(d.tool_calls||[]).map(t=>`<div class="card mono">${esc(t.tool)} ${esc(JSON.stringify(t.arguments||{}))}${t.result&&t.result.error?`<div style="color:var(--red)">${esc(t.result.error)}</div>`:''}</div>`).join(''); chatMsgs.innerHTML+=`<div class="msg"><div class="bubble">${tools}${md(d.reply)}${src}</div></div>`} chatMsgs.scrollTop=chatMsgs.scrollHeight}
async function switchModel(v){currentModel=v; await api('/model',{model:v}); log(`已切换模型到 ${v}`)}
async function runSearch(){if(!activeProject)return alert('请先选择工程'); const q=searchQ.value.trim(); if(!q)return; const d=await api('/search',{project_id:activeProject,query:q,domain:searchDomain.value||null,path:searchPath.value.trim()||null,top_k:parseInt(searchTopK.value||'8',10)}); if(d.error){searchR.innerHTML=`<div class="muted" style="color:var(--red)">${esc(d.error)}</div>`; return} pushSuggested((d.results||[]).map(r=>r.path)); searchR.innerHTML=!d.results||!d.results.length?'<div class="muted">没有匹配结果</div>':d.results.map(r=>`<div class="card" onclick="selectedFile='${r.path.replace(/'/g,\"\\\\'\")}'; readPath.value='${r.path.replace(/'/g,\"\\\\'\")}'; tab('files',document.querySelectorAll('.tab')[2]); readFile();"><div><span style="color:var(--accent)">${esc(r.path)}</span> <span class="muted">${esc(r.chunk_type)} ${r.collection_domain?`· ${esc(r.collection_domain)}`:''} · ${Number(r.score||0).toFixed(3)}</span></div><div class="mono">${esc(r.preview||'')}</div></div>`).join('')}
async function listFiles(){if(!activeProject)return alert('请先选择工程'); const d=await api('/files/list',{project_id:activeProject,path:lsPath.value.trim()||'.',pattern:lsPattern.value.trim()||'*'}); if(d.error){filesR.innerHTML=`<div class="muted" style="color:var(--red)">${esc(d.error)}</div>`; return} filesR.innerHTML=!d.items||!d.items.length?'<div class="muted">目录为空</div>':d.items.map(i=>`<div class="card" onclick="selectedFile='${i.path.replace(/'/g,\"\\\\'\")}'; readPath.value='${i.path.replace(/'/g,\"\\\\'\")}';"><div><span style="color:var(--accent)">${esc(i.path)}</span></div><div class="muted">${esc(i.type)} · ${i.size}</div></div>`).join('')}
async function readFile(){if(!activeProject)return alert('请先选择工程'); const p=readPath.value.trim()||selectedFile; if(!p)return; const d=await api('/files/read',{project_id:activeProject,path:p,max_lines:parseInt(readLines.value||'200',10)}); lastReadContent=d.error?'':(d.content||''); editorR.value=d.error?d.error:lastReadContent; diffR.textContent='Diff 预览会显示在这里'}
async function previewDiff(){if(!activeProject)return alert('请先选择工程'); const p=readPath.value.trim()||selectedFile; if(!p)return; const d=await api('/files/diff',{project_id:activeProject,path:p,proposed_content:editorR.value}); diffR.textContent=d.error?d.error:(!d.changed?'没有变更':(d.diff||'没有变更'))}
async function saveFile(){if(!activeProject)return alert('请先选择工程'); const p=readPath.value.trim()||selectedFile; if(!p)return; const d=await api('/files/save',{project_id:activeProject,path:p,content:editorR.value}); if(d.error){diffR.textContent=d.error; return} lastReadContent=editorR.value; diffR.textContent='保存成功'; pushSuggested([p]); log(`已保存 ${p}`); await loadDrafts()}
async function draftEdit(){if(!activeProject)return alert('请先选择工程'); const p=readPath.value.trim()||selectedFile; const inst=editPrompt.value.trim(); if(!p)return alert('请先选择文件'); if(!inst)return alert('请输入修改指令'); diffR.textContent='AI 正在生成草案...'; const d=await api('/edit/draft',{project_id:activeProject,path:p,instruction:inst,model:currentModel}).catch(e=>({error:e.message})); if(d.error){diffR.textContent=d.error; return} editorR.value=d.proposed_content||editorR.value; diffR.textContent=d.changed?(d.diff||'没有变更'):'没有变更'; pushSuggested([p]); await loadDrafts(); log(`AI 已生成草案: ${p} (${d.model||currentModel})`)}
function revertDraft(){editorR.value=lastReadContent; diffR.textContent='已丢弃草案'; log('已丢弃当前草案')}
function useEditorInChat(){const p=readPath.value.trim()||selectedFile; if(!p)return; chatPath.value=p; tab('chat',document.querySelectorAll('.tab')[0]); log(`已将 ${p} 加入聊天路径过滤`)}
async function grepFiles(){if(!activeProject)return alert('请先选择工程'); const q=grepQ.value.trim(); if(!q)return; const d=await api('/files/grep',{project_id:activeProject,query:q,path:lsPath.value.trim()||'.',pattern:grepPattern.value.trim()||'*.cs'}); if(d.error){grepR.innerHTML=`<div class="muted" style="color:var(--red)">${esc(d.error)}</div>`; return} grepR.innerHTML=!d.results||!d.results.length?'<div class="muted">没有匹配结果</div>':d.results.map(r=>`<div class="card" onclick="selectedFile='${r.file.replace(/'/g,\"\\\\'\")}'; readPath.value='${r.file.replace(/'/g,\"\\\\'\")}'; readFile();"><div><span style="color:var(--accent)">${esc(r.file)}</span> <span class="muted">line ${r.line}</span></div><div class="mono">${esc(r.content)}</div></div>`).join('')}
async function loadStats(){if(!activeProject)return; const d=await api(`/stats?project_id=${encodeURIComponent(activeProject)}`,null,'GET'); if(d.error)return; const domains=Object.entries(d.domain_distribution||{}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${esc(k)}: ${v}`).join('<br>')||'暂无'; const types=Object.entries(d.type_distribution||{}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${esc(k)}: ${v}`).join('<br>')||'暂无'; statsR.innerHTML=`<div class="grid"><div class="stat"><div class="muted">Chunks</div><div class="value green">${d.total_chunks}</div></div><div class="stat"><div class="muted">文件数</div><div class="value blue">${d.file_count}</div></div><div class="stat"><div class="muted">集合</div><div>${esc(d.collection_name)}</div></div><div class="stat"><div class="muted">数据库</div><div class="muted">${esc(d.db_path)}</div></div></div><div class="card"><b>域分布</b><div class="mono">${domains}</div></div><div class="card"><b>Chunk 类型分布</b><div class="mono">${types}</div></div>`}
async function refreshAll(){await loadProjects();await loadConvs();await loadStats();await loadDrafts();await loadTasks();renderSuggestedFiles()}
refreshAll();
</script></body></html>
"""


def _get_project(project_id: str | None):
    pm = get_project_manager()
    return pm.get_project(project_id) if project_id else pm.get_active()


def _get_or_create_agent(project_id: str) -> UnityAgent:
    if project_id and project_id not in _agents:
        pm = get_project_manager()
        project = pm.get_project(project_id)
        provider = create_provider()
        retriever = Retriever(
            db_path=project.db_path if project else config.CHROMA_DB_ROOT,
            collection_name=project.collection_name if project else "unity_codebase",
        )
        _agents[project_id] = UnityAgent(project=project, provider=provider, retriever=retriever)
    return _agents.get(project_id) or _agents.setdefault("_default", UnityAgent())


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    match = re.match(r"^```[\w+-]*\n([\s\S]*?)\n```$", cleaned)
    return match.group(1) if match else cleaned


def _draft_bucket(project_id: str) -> list[dict]:
    return _drafts.setdefault(project_id or "_global", [])


def _task_bucket(project_id: str) -> list[dict]:
    return _tasks.setdefault(project_id or "_global", [])


def _generate_edit_draft(project, path: str, instruction: str, model: str = "") -> dict:
    read_result = tool_read_file(project.path, path, max_lines=5000)
    if read_result.get("error"):
        return {"error": read_result["error"]}

    model_id = model or (config.OPENAI_MODEL if config.LLM_PROVIDER == "openai" else config.CLAUDE_MODEL)
    try:
        _, provider = provider_for_model(model_id)
    except Exception:
        provider = create_provider(model=model_id)

    system_prompt = (
        "You are editing a single source file in a local codebase. "
        "Return only the full updated file contents, with no explanation, no markdown fences, and no surrounding commentary. "
        "Preserve unrelated code and formatting unless the instruction requires a change."
    )
    user_prompt = (
        f"Project: {project.name}\n"
        f"Path: {path}\n"
        f"Instruction:\n{instruction}\n\n"
        "Current file contents:\n"
        f"{read_result['content']}"
    )
    result = provider.chat(
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=None,
        max_tokens=6000,
    )
    proposed = _strip_code_fences(result.text)
    if not proposed:
        return {"error": "模型没有返回可用的文件内容"}

    diff_result = tool_diff_file(project.path, path, proposed)
    if diff_result.get("error"):
        return {"error": diff_result["error"]}

    return {
        "path": path,
        "model": provider.model_name(),
        "proposed_content": proposed,
        "diff": diff_result.get("diff", ""),
        "changed": diff_result.get("changed", False),
    }


def _create_draft(project_id: str, draft: dict, instruction: str) -> dict:
    item = {
        "draft_id": uuid.uuid4().hex[:10],
        "path": draft["path"],
        "instruction": instruction,
        "model": draft.get("model", ""),
        "proposed_content": draft.get("proposed_content", ""),
        "diff": draft.get("diff", ""),
        "changed": draft.get("changed", False),
        "created_at": time.time(),
    }
    bucket = _draft_bucket(project_id)
    bucket.insert(0, item)
    return item


def _apply_draft(project, draft: dict) -> dict:
    return tool_write_file(project.path, draft["path"], draft["proposed_content"])


@app.route("/")
def index():
    current = config.OPENAI_MODEL if config.LLM_PROVIDER == "openai" else config.CLAUDE_MODEL
    return render_template_string(DASHBOARD_HTML, models=get_all_models(), current_model=current)


@app.route("/api/projects")
def api_projects():
    pm = get_project_manager(); active = pm.get_active()
    projects = [{**p.to_dict(), "is_active": active and p.project_id == active.project_id} for p in pm.list_projects()]
    return jsonify({"projects": projects, "active_id": active.project_id if active else None})


@app.route("/api/projects/add", methods=["POST"])
def api_add_project():
    d = request.json; pm = get_project_manager()
    try:
        p = pm.add_project(path=d["path"], name=d.get("name", ""), unity_version=d.get("unity_version", "2022.3 LTS"), render_pipeline=d.get("render_pipeline", "URP"))
        pm.set_active(p.project_id); _agents.pop(p.project_id, None)
        return jsonify({"ok": True, "project_id": p.project_id, "name": p.name})
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)})


@app.route("/api/projects/active", methods=["POST"])
def api_set_active():
    d = request.json; pm = get_project_manager()
    ok = pm.set_active(d["project_id"]); _agents.pop(d["project_id"], None)
    return jsonify({"ok": ok})


@app.route("/api/stats")
def api_stats():
    project = _get_project(request.args.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    return jsonify(Retriever(db_path=project.db_path, collection_name=project.collection_name).get_stats())


@app.route("/api/index/build", methods=["POST"])
def api_build_index():
    global _index_running
    if _index_running: return jsonify({"error": "索引正在执行"})
    project = _get_project(request.json.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    _index_running = True
    try:
        chunks, files = build_index(project.path, project.db_path, project.collection_name, project.project_id)
        _agents.pop(project.project_id, None)
        return jsonify({"ok": True, "chunks": chunks, "files": files})
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        _index_running = False


@app.route("/api/index/update", methods=["POST"])
def api_update_index():
    project = _get_project(request.json.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    try:
        update_index(project.path, project.db_path, project.collection_name, project.project_id)
        _agents.pop(project.project_id, None)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/conversations")
def api_conversations():
    cm = ConversationManager(request.args.get("project_id", "_global"))
    convs = [{"conv_id": c.conv_id, "title": c.title, "model": c.model, "message_count": c.message_count, "updated_at": c.updated_at} for c in cm.list_conversations()]
    return jsonify({"conversations": convs})


@app.route("/api/conversations/new", methods=["POST"])
def api_new_conv():
    d = request.json; agent = _get_or_create_agent(d.get("project_id", "_global")); model = d.get("model", "")
    if model and model != agent.provider.model_name():
        try: agent.switch_model(model)
        except Exception: pass
    return jsonify({"ok": True, "conv_id": agent.start_new_conversation()})


@app.route("/api/conversations/load", methods=["POST"])
def api_load_conv():
    d = request.json; agent = _get_or_create_agent(d.get("project_id", "_global"))
    if not agent.load_conversation(d["conv_id"]): return jsonify({"error": "对话不存在"})
    conv = agent._current_conv
    return jsonify({"ok": True, "messages": conv.messages, "model": conv.meta.model, "title": conv.meta.title})


@app.route("/api/conversations/delete", methods=["POST"])
def api_delete_conv():
    d = request.json
    return jsonify({"ok": _get_or_create_agent(d.get("project_id", "_global")).delete_conversation(d["conv_id"])})


@app.route("/api/conversations/rename", methods=["POST"])
def api_rename_conv():
    d = request.json; title = d.get("title", "").strip()
    if not title: return jsonify({"error": "标题不能为空"})
    return jsonify({"ok": _get_or_create_agent(d.get("project_id", "_global")).rename_conversation(d["conv_id"], title)})


@app.route("/api/model", methods=["POST"])
def api_model():
    model = request.json.get("model", "")
    for a in _agents.values():
        try: a.switch_model(model)
        except Exception: pass
    return jsonify({"ok": True, "model": model})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    d = request.json; msg = d.get("message", "").strip()
    if not msg: return jsonify({"error": "消息不能为空"})
    agent = _get_or_create_agent(d.get("project_id"))
    model = d.get("model")
    if model and model != agent.provider.model_name():
        try: agent.switch_model(model)
        except Exception: pass
    try:
        reply, results, tools = agent.chat(msg, top_k=d.get("top_k"), filter_path=d.get("filter_path"), filter_domain=d.get("filter_domain"))
        stats = agent.get_token_stats()
        sources = [{"path": r.relative_path, "score": r.score, "collection_domain": getattr(r, "collection_domain", "")} for r in results]
        return jsonify({"reply": reply, "sources": sources, "tool_calls": tools, "model": stats["model"], "conv_id": stats.get("conv_id")})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/search", methods=["POST"])
def api_search():
    d = request.json; project = _get_project(d.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    query = d.get("query", "").strip()
    if not query: return jsonify({"error": "查询不能为空"})
    results = Retriever(db_path=project.db_path, collection_name=project.collection_name).search(query, top_k=int(d.get("top_k") or config.RAG_TOP_K), filter_path=d.get("path"), filter_domain=d.get("domain"))
    return jsonify({"results": [{"path": r.relative_path, "chunk_type": r.chunk_type, "collection_domain": getattr(r, "collection_domain", ""), "score": r.score, "preview": r.text[:500]} for r in results]})


@app.route("/api/files/list", methods=["POST"])
def api_files_list():
    d = request.json; project = _get_project(d.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    return jsonify(tool_list_files(project.path, path=d.get("path", "."), pattern=d.get("pattern", "*")))


@app.route("/api/files/read", methods=["POST"])
def api_files_read():
    d = request.json; project = _get_project(d.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    return jsonify(tool_read_file(project.path, d.get("path", ""), max_lines=int(d.get("max_lines") or 200)))


@app.route("/api/files/diff", methods=["POST"])
def api_files_diff():
    d = request.json; project = _get_project(d.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    return jsonify(tool_diff_file(project.path, d.get("path", ""), d.get("proposed_content", ""), context_lines=int(d.get("context_lines") or 3)))


@app.route("/api/files/save", methods=["POST"])
def api_files_save():
    d = request.json; project = _get_project(d.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    return jsonify(tool_write_file(project.path, d.get("path", ""), d.get("content", "")))


@app.route("/api/edit/draft", methods=["POST"])
def api_edit_draft():
    d = request.json; project = _get_project(d.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    path = (d.get("path") or "").strip()
    instruction = (d.get("instruction") or "").strip()
    if not path: return jsonify({"error": "文件路径不能为空"})
    if not instruction: return jsonify({"error": "修改指令不能为空"})
    draft = _generate_edit_draft(project, path, instruction, model=d.get("model", ""))
    if draft.get("error"):
        return jsonify(draft)
    item = _create_draft(d.get("project_id") or project.project_id, draft, instruction)
    return jsonify(item)


@app.route("/api/edit/drafts")
def api_edit_drafts():
    project_id = request.args.get("project_id", "_global")
    drafts = [{
        "draft_id": item["draft_id"],
        "path": item["path"],
        "instruction": item["instruction"],
        "model": item["model"],
        "changed": item["changed"],
        "created_at": item["created_at"],
        "diff": item["diff"],
        "proposed_content": item["proposed_content"],
    } for item in _draft_bucket(project_id)]
    return jsonify({"drafts": drafts})


@app.route("/api/edit/apply", methods=["POST"])
def api_edit_apply():
    d = request.json; project = _get_project(d.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    bucket = _draft_bucket(d.get("project_id") or project.project_id)
    draft = next((item for item in bucket if item["draft_id"] == d.get("draft_id")), None)
    if not draft: return jsonify({"error": "未找到 draft"})
    result = _apply_draft(project, draft)
    if not result.get("success"): return jsonify(result)
    bucket[:] = [item for item in bucket if item["draft_id"] != draft["draft_id"]]
    return jsonify({"ok": True, "path": draft["path"]})


@app.route("/api/edit/apply_all", methods=["POST"])
def api_edit_apply_all():
    d = request.json; project = _get_project(d.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    bucket = _draft_bucket(d.get("project_id") or project.project_id)
    applied = []
    failed = []
    for draft in list(bucket):
        result = _apply_draft(project, draft)
        if result.get("success"):
            applied.append(draft["path"])
            bucket.remove(draft)
        else:
            failed.append({"path": draft["path"], "error": result.get("error", "未知错误")})
    return jsonify({"ok": True, "applied": applied, "failed": failed})


@app.route("/api/edit/discard", methods=["POST"])
def api_edit_discard():
    d = request.json; project_id = d.get("project_id", "_global")
    bucket = _draft_bucket(project_id)
    before = len(bucket)
    bucket[:] = [item for item in bucket if item["draft_id"] != d.get("draft_id")]
    return jsonify({"ok": len(bucket) != before})


@app.route("/api/edit/discard_all", methods=["POST"])
def api_edit_discard_all():
    d = request.json; project_id = d.get("project_id", "_global")
    bucket = _draft_bucket(project_id)
    count = len(bucket)
    bucket.clear()
    return jsonify({"ok": True, "discarded": count})


@app.route("/api/tasks")
def api_tasks():
    project_id = request.args.get("project_id", "_global")
    return jsonify({"tasks": _task_bucket(project_id)})


@app.route("/api/tasks/add", methods=["POST"])
def api_task_add():
    d = request.json; project_id = d.get("project_id", "_global")
    text = (d.get("text") or "").strip()
    if not text: return jsonify({"error": "任务内容不能为空"})
    item = {"task_id": uuid.uuid4().hex[:10], "text": text, "done": False, "created_at": time.time()}
    bucket = _task_bucket(project_id); bucket.insert(0, item)
    return jsonify({"ok": True, "task": item})


@app.route("/api/tasks/toggle", methods=["POST"])
def api_task_toggle():
    d = request.json; project_id = d.get("project_id", "_global")
    for item in _task_bucket(project_id):
        if item["task_id"] == d.get("task_id"):
            item["done"] = not item.get("done", False)
            return jsonify({"ok": True, "task": item})
    return jsonify({"error": "未找到任务"})


@app.route("/api/tasks/delete", methods=["POST"])
def api_task_delete():
    d = request.json; project_id = d.get("project_id", "_global")
    bucket = _task_bucket(project_id)
    before = len(bucket)
    bucket[:] = [item for item in bucket if item["task_id"] != d.get("task_id")]
    return jsonify({"ok": len(bucket) != before})


@app.route("/api/files/grep", methods=["POST"])
def api_files_grep():
    d = request.json; project = _get_project(d.get("project_id"))
    if not project: return jsonify({"error": "未找到工程"})
    return jsonify(tool_search_code(project.path, query=d.get("query", ""), path=d.get("path", "."), file_pattern=d.get("pattern", "*.cs"), max_results=50))


def run_dashboard(host=None, port=None, debug=False):
    host = host or config.DASHBOARD_HOST; port = port or config.DASHBOARD_PORT
    print(f"\n  Unity RAG Agent Dashboard\n  http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug)
