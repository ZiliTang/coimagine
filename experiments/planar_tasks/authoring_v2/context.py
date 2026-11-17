"""Lossy, explicitly marked context view; original tool evidence remains retrievable."""
from copy import deepcopy
import hashlib,json


def size(x):return len(json.dumps(x,ensure_ascii=False).encode())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def project(messages,settings,notebook=''):
    original=size(messages);starts=[i for i,m in enumerate(messages) if m['role']=='assistant']
    note=dict(role='user',content='HOST NOTEBOOK (author-written, not verified facts):\n'+notebook) if notebook else None
    if original<=settings['context_trigger_bytes']:
        out=deepcopy(messages)
        if note:out.insert(2,note)
        return out,dict(compacted=False,original_bytes=original,projected_bytes=size(out),original_sha256=digest(messages),projected_sha256=digest(out))
    for recent in range(min(settings['recent_turns'],len(starts)),-1,-1):
        cutoff=starts[-recent] if recent else len(messages)
        archived=messages[2:cutoff];facts=[];records=[]
        for m in archived:
            if m['role']!='tool':continue
            try:r=json.loads(m['content'])
            except (ValueError,TypeError):continue
            ident=r.get('record_id')
            if ident is not None:records.append(ident)
            if r.get('ok') is False:
                facts.append(dict(record_id=ident,ok=False,error=str(r.get('error'))[:400]))
            else:
                result=r.get('result')
                if isinstance(result,dict) and ('test_id' in result or 'batch_id' in result):
                    facts.append(dict(record_id=ident,test_id=result.get('test_id'),batch_id=result.get('batch_id'),
                                      summary=result.get('summary'),status=result.get('status')))
        # This is an evidence index, not an LLM-generated narrative or a claim to retain all reasoning.
        index=dict(kind='HOST_CONTEXT_ARCHIVE_NOTICE',archived_messages=len(archived),
                   record_range=[min(records),max(records)] if records else None,
                   facts_tail=facts[-12:],omitted_fact_count=max(0,len(facts)-12),
                   retrieval='list_results(start,count), then read_result(record_id,start,count). Original requests/results remain on disk. Prior reasoning/code text is not retained in this context view; use current files and author notebook.',
                   finalization='Only submit is available if the current budget says submission phase.')
        out=deepcopy(messages[:2])+[dict(role='user',content=json.dumps(index,ensure_ascii=False))]
        if note:out.append(note)
        out+=deepcopy(messages[cutoff:])
        if size(out)<=settings['context_max_bytes']:
            return out,dict(compacted=True,original_bytes=original,projected_bytes=size(out),original_sha256=digest(messages),projected_sha256=digest(out),retained_recent_turns=recent,archived_messages=len(archived),record_range=index['record_range'])
    raise ValueError('fixed instructions/notebook exceed context cap; no silent truncation')
