#!/usr/bin/env python3
"""AAAI-27 literature survey: online 3D instance segmentation x foundation features.
Batch arXiv API queries; compact structured dump for synthesis."""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time

NS = {'a': 'http://www.w3.org/2005/Atom'}


def q(query, n=10, sort='relevance'):
    url = ('https://export.arxiv.org/api/query?search_query=' + urllib.parse.quote(query)
           + f'&max_results={n}&sortBy={sort}&sortOrder=descending')
    for attempt in range(3):
        try:
            xml = urllib.request.urlopen(url, timeout=25).read().decode()
            break
        except Exception as e:
            if attempt == 2:
                return [('QUERY-FAILED', str(e)[:60], '', '')]
            time.sleep(4)
    out = []
    for e in ET.fromstring(xml).findall('a:entry', NS):
        t = ' '.join(e.find('a:title', NS).text.split())
        aid = e.find('a:id', NS).text.split('/abs/')[-1]
        pub = e.find('a:published', NS).text[:10]
        summ = ' '.join(e.find('a:summary', NS).text.split())[:240]
        out.append((t, aid, pub, summ))
    time.sleep(3.2)
    return out


BATTERY = [
    # -- A. online lineage pinning --
    ('A1', 'all:"EmbodiedSAM"', 5, 'relevance'),
    ('A2', 'all:"ESAM++"', 3, 'relevance'),
    ('A3', 'all:"online 3D instance segmentation"', 15, 'relevance'),
    ('A4', 'all:"online 3D instance segmentation"', 15, 'submittedDate'),
    ('A5', 'abs:"streaming" AND abs:"3D instance segmentation"', 10, 'submittedDate'),
    ('A6', 'all:"SAI3D" OR all:"SAMPro3D" OR all:"SAM3D"', 10, 'relevance'),
    ('A7', 'abs:"online" AND abs:"point cloud" AND abs:"instance segmentation"', 15, 'submittedDate'),
    # -- B. gap scan: foundation features x 3D --
    ('B1', 'abs:"DINOv2" AND abs:"3D" AND abs:"segmentation"', 15, 'submittedDate'),
    ('B2', 'abs:"foundation model" AND abs:"3D instance segmentation"', 15, 'submittedDate'),
    ('B3', 'abs:"2D features" AND abs:"point cloud" AND abs:"segmentation"', 12, 'submittedDate'),
    ('B4', 'all:"DINO in the Room"', 3, 'relevance'),
    ('B5', 'all:"SegDINO3D"', 3, 'relevance'),
    ('B6', 'abs:"vision foundation model" AND abs:"online" AND abs:"3D"', 12, 'submittedDate'),
    # -- C. offline instance seg SOTA line --
    ('C1', 'all:"Mask3D" AND cat:cs.CV', 5, 'relevance'),
    ('C2', 'all:"OneFormer3D" OR all:"SPFormer" OR all:"superpoint transformer"', 8, 'relevance'),
    # -- D. 2D->3D injection / distillation line --
    ('D1', 'abs:"distill" AND abs:"2D" AND abs:"3D segmentation"', 12, 'submittedDate'),
    ('D2', 'all:"BPNet" OR all:"DeepViewAgg" OR all:"ODIN" AND abs:"3D segmentation"', 8, 'relevance'),
    ('D3', 'all:"OpenScene" OR all:"OpenMask3D" OR all:"Open3DIS"', 8, 'relevance'),
    # -- E. concurrency scan (fresh, past 12 months emphasis) --
    ('E1', 'cat:cs.CV AND abs:"online" AND abs:"RGB-D" AND abs:"segmentation"', 15, 'submittedDate'),
    ('E2', 'abs:"embodied" AND abs:"3D segmentation"', 12, 'submittedDate'),
]

for tag, query, n, sort in BATTERY:
    print(f'\n##### {tag} [{sort}] {query}')
    for t, aid, pub, summ in q(query, n, sort):
        print(f'- {pub} | {aid} | {t}')
        print(f'    {summ}')
