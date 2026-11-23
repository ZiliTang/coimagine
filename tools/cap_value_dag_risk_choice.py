"""Exploratory risk-based routing using the unchanged five terminal moment forecasts."""
import numpy as np

from tools import cap_value_dag as f
from tools.cap_value_dag_report import stat


def evaluate(fold):
    cases,pred=f.evaluate(fold)
    p=pred['public_history/root']
    y=np.array([[c['branches'][r]['y'] for r in f.ROUTES] for c in cases])
    ids=np.array([c['room'] for c in cases])
    modes={'minimum_expected_distance':p[:,:,0],'minimum_tail_probability':p[:,:,3],
           'minimum_mean_plus_sd':p[:,:,0]+np.sqrt(f.variance(p))}
    result={}
    for k,scores in modes.items():
        idx=np.nan_to_num(scores,nan=1e8).round(12).argmin(1)
        act=y[np.arange(len(cases)),idx]
        result[k]=dict(success=int(act[:,2].sum()),tail=int(act[:,3].sum()),ref_tail=int(y[:,0,3].sum()),
                       gap=float(act[:,0].mean()),ref_success=int(y[:,0,2].sum()),
                       tail_delta=stat(act[:,3]-y[:,0,3],ids),success_delta=stat(act[:,2]-y[:,0,2],ids),
                       recover=int(sum(act[:,2]>y[:,0,2])),loss=int(sum(act[:,2]<y[:,0,2])))
    return result


if __name__=='__main__':
    f.save(f.DOC/'risk_choice.json',{fold:evaluate(fold) for fold in ('diagnostic','confirm')})
