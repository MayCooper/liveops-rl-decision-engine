CREATE TABLE IF NOT EXISTS `PROJECT_ID.liveops_policy_lab.simulated_player_episodes` (
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  player_id INT64,
  day INT64,
  policy_name STRING,
  row_json STRING
);

CREATE TABLE IF NOT EXISTS `PROJECT_ID.liveops_policy_lab.policy_eval_results` (
  created_at TIMESTAMP,
  metrics_json STRING
);

CREATE TABLE IF NOT EXISTS `PROJECT_ID.liveops_policy_lab.recommendation_logs` (
  created_at TIMESTAMP,
  request_json STRING,
  response_json STRING
);

CREATE TABLE IF NOT EXISTS `PROJECT_ID.liveops_policy_lab.agent_audit_logs` (
  created_at TIMESTAMP,
  report_json STRING
);
