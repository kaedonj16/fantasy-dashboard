from dashboard_services.data_building.value_model_training import CURRENT_SEASON
from dashboard_services.data_building.value_model_training import (
    load_trained_bundle,
    build_ml_value_table,
)

bundle = load_trained_bundle()
values = build_ml_value_table(CURRENT_SEASON, weeks=range(1, 19))

bundle = load_trained_bundle()


# Top 20 by value
top20 = sorted(values.items(), key=lambda kv: kv[1], reverse=True)[:20]
for pid, val in top20:
    print(pid, val)
