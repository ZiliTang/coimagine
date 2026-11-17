// The server rejects all mutation methods; these guards also remove editing affordances.
for(const id of ['resetPoses','runBtn','addSegment','clearSequence','savePreset','saveOverrides'])document.getElementById(id).disabled=true;
const status=document.getElementById('status');new MutationObserver(()=>{if(status.textContent.includes('save as a new preset'))status.textContent=status.textContent.replace(' — save as a new preset if useful',' · 只读回放');}).observe(status,{childList:true});
document.title='CoImagine · Read-only Research Replay';
