"""Independent properties of the paper's equations, not a test of the search engine."""
import math

def scalar(c,p,cmin,catp,pmin,patc,lam,rho=1e-4):
    dc=(c-cmin)/(catp-cmin);dp=(p-pmin)/(patc-pmin)
    return max((1-lam)*dc,lam*dp)+rho*((1-lam)*dc+lam*dp)

def weighted_latency(duration,priority,route):
    arrivals={};t=0;last='depot'
    for j in route:
        t+=duration[last,j];arrivals[j]=t;last=j
    return sum(arrivals[j]*priority[j] for j in route)

def flow_latency(duration,priority,route):
    prev='depot';total=0
    for idx,j in enumerate(route):
        remaining=sum(priority[x] for x in route[idx:])
        total+=duration[prev,j]*remaining;prev=j
    return total

def test_priority_flow_equivalence():
    duration={('depot','a'):4,('a','b'):7,('b','c'):6}
    priority={'a':1.0,'b':0.4,'c':10.0}
    assert math.isclose(weighted_latency(duration,priority,['a','b','c']),flow_latency(duration,priority,['a','b','c']))

def test_positive_scalar_monotonicity():
    assert scalar(120,100,100,200,50,150,0.5) < scalar(130,120,100,200,50,150,0.5)
