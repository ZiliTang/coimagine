import json,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path('/home/zilitang/repo/coimagine')
report=root/'docs/research/reports/CoImagine_Progress_20261117/index.html'
out=root/'outputs/research/supervisor_report_20261117/latest'
out.mkdir(exist_ok=True)
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
 errors=[];external=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http:','https:')) else None)
 page.goto(report.as_uri());page.wait_for_timeout(400)
 assert page.title().startswith('CoImagine')
 assert page.locator('#scene-grid canvas').count()==8
 assert page.locator('#source-list .source-item').count()>=16
 assert page.locator('#random-scene-grid canvas').count()==4
 data=page.locator('#report-data').text_content();embedded=json.loads(data)
 assert '270' in page.locator('#latest').inner_text()
 assert len(embedded['replays'])>=10
 page.screenshot(path=str(out/'overview_desktop.png'))
 page.locator('#sample-switch button[data-n="150"]').click()
 assert page.locator('#attribution-chart').inner_text().count('124/216')==2
 page.locator('#split-switch button[data-split="transfer"]').click()
 assert '7/48' in page.locator('#transfer-chart').inner_text()
 assert page.locator('#transfer-chart').inner_text().count('1/48')==4
 for key,n in [(key,len(indices)) for key,indices in dict(clear=[0],wall=[1],integration=[2,3],transfer=[4,5],**embedded['extra_examples']).items()]:
  page.select_option('#example',key);assert page.locator('#players canvas').count()==n
  page.locator('#next-boundary').click();assert page.locator('#time').inner_text()!='0.00 s'
  page.locator('#reset').click();assert page.locator('#time').inner_text()=='0.00 s'
  page.locator('#play').click();page.wait_for_timeout(400);page.locator('#play').click();assert float(page.locator('#time').inner_text().split()[0])>.1
 page.select_option('#example','repair');page.locator('#handoff').click();assert '接管之后' in page.locator('#action-0').inner_text()
 page.locator('#seek').evaluate('(e)=>{e.value=30;e.dispatchEvent(new Event("input",{bubbles:true}));}')
 assert page.locator('#time').inner_text()=='30.00 s'
 page.locator('#replay').scroll_into_view_if_needed();page.wait_for_timeout(250);page.screenshot(path=str(out/'replay_desktop.png'))
 page.locator('#latest').scroll_into_view_if_needed();page.screenshot(path=str(out/'latest_desktop.png'))
 desktop_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(250)
 mobile_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
 assert not desktop_overflow and not mobile_overflow,(desktop_overflow,mobile_overflow)
 page.locator('#replay').scroll_into_view_if_needed();page.wait_for_timeout(250);page.screenshot(path=str(out/'replay_mobile.png'))
 page.locator('#latest').scroll_into_view_if_needed();page.screenshot(path=str(out/'latest_mobile.png'))
 page.locator('#overview').scroll_into_view_if_needed();page.screenshot(path=str(out/'overview_mobile.png'))
 page.set_viewport_size({'width':1440,'height':1000});page.select_option('#example','clear');page.locator('#seek').evaluate('(e)=>{e.value=20;e.dispatchEvent(new Event("input",{bubbles:true}));}');page.wait_for_timeout(150)
 page.pdf(path=str(out/'CoImagine_Progress_20261117.pdf'),format='A4',print_background=True,margin={'top':'12mm','bottom':'12mm','left':'12mm','right':'12mm'})
 (report.parent/'CoImagine_Progress_20261117.pdf').write_bytes((out/'CoImagine_Progress_20261117.pdf').read_bytes())
 assert not errors,errors
 assert not external,external
 result=dict(report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),status='passed',page_errors=errors,external_requests=external,scenes=12,replays=len(embedded['replays']),interactive_cases=4+len(embedded['extra_examples']),desktop=[1440,1000],mobile=[390,844],horizontal_overflow=False,checks=['offline file loading','chart toggles','case selection','play/pause','seek','decision jumps','reset','responsive layout','PDF export'])
 (out/'browser_check.json').write_text(json.dumps(result,indent=2))
 print(result)
 browser.close()
