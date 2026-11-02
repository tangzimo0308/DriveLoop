"""Summarize all attempts without selecting a different output or dropping failures."""
import hashlib
import json
import sys
from pathlib import Path


def summarize(root):
    report = {'protocol': str(root / 'protocol.json'), 'arms': {},
              'claim_boundary': 'One source window, one seed bank; detector proxy is not human-verified cut-in success.'}
    for arm in ('direct', 'fixed', 'full'):
        result = json.loads((root / arm / 'result.json').read_text())
        attempts = result['attempt_history']
        assert len(attempts) == 4
        rows = []
        for attempt in attempts:
            generation = attempt['generation']
            evaluation = attempt['evaluation']
            video = Path(generation['artifacts']['video'])
            assert video.is_file()
            mapping = generation['metadata'].get('dd2_override_json', {}).get('audit', {}).get('actor_motion_frame_mapping', {})
            rows.append({'iteration': attempt['iteration'], 'status': attempt['status'],
                         'evaluation': evaluation, 'trajectory_mapping': mapping,
                         'video': str(video), 'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest()})
        report['arms'][arm] = {'attempts': rows, 'return_status': result['return_status'],
                              'returned_iteration': result['returned_generation']['iteration'],
                              'returned_evaluation': result['returned_evaluation']}
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    r = summarize(Path(sys.argv[1]).resolve())
    print(json.dumps({k: {'return_status': v['return_status'], 'returned_iteration': v['returned_iteration'],
                        'S_perc': [a['evaluation']['metrics'].get('S_perc') for a in v['attempts']]}
                      for k, v in r['arms'].items()}, indent=2))
