import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')

if os.path.exists('state/learning_journal.json'):
    os.remove('state/learning_journal.json')

t0 = time.time()
from analytics.learning_engine import seed_from_history
j = seed_from_history()
elapsed = time.time() - t0

meta      = j.get('metadata', {})
patterns  = j.get('patterns', {})
archetypes= j.get('archetypes', {})

print(f"\n{'='*55}")
print(f"  SEEDING COMPLETE in {elapsed:.1f}s")
print(f"  Unique pattern keys : {len(patterns)}")
print(f"  Archetypes          : {len(archetypes)}")
print(f"  Total observations  : {meta.get('total_observations', 0)}")
print(f"{'='*55}")

# State breakdown
states = {}
for p in patterns.values():
    s = p.get('state', 'WATCHING')
    states[s] = states.get(s, 0) + 1
print("\nPattern states:")
for s, c in sorted(states.items(), key=lambda x: -x[1]):
    print(f"  {s:12s}: {c}")

# Archetype breakdown
print("\nArchetype totals (decayed trades):")
for name, data in sorted(archetypes.items(), key=lambda x: -x[1].get('trades', 0)):
    wr = data.get('win_rate', 0)
    tr = data.get('trades', 0)
    print(f"  {name:30s}: {tr:7.1f} trades  WR={wr:.1%}")
