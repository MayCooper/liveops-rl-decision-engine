# Cold-start Policy

Cold-start users have little or no behavioral history. The system should prefer safe onboarding actions, such as training matches or doing nothing, until there is enough history to estimate player pressure more confidently.

Cold-start users should not receive elite difficulty, repeated grants, or aggressive difficulty changes. The simulator exposes `cold_start` and `history_confidence` so the UI can show why the served action becomes more conservative.
