-- Historical prospect grades for draft classes 2018-2025.
-- Stores simplified pre-draft grades for use as comparables in the prospect modal.

CREATE TABLE IF NOT EXISTS historical_prospect_grades (
    player_id           TEXT        PRIMARY KEY,  -- HIST_{YEAR}_{NAME_SLUG}
    sleeper_id          TEXT,
    name                TEXT        NOT NULL,
    position            TEXT        NOT NULL,
    draft_class_year    INTEGER     NOT NULL,
    school              TEXT,
    prospect_score      DECIMAL(6,2),
    tier                INTEGER,
    tier_label          TEXT,
    overall_rank        INTEGER,
    position_rank       INTEGER,
    actual_pick         INTEGER,
    actual_round        INTEGER,
    actual_nfl_team     TEXT,
    production_score    DECIMAL(6,2),
    athleticism_score   DECIMAL(6,2),
    draft_capital_score DECIMAL(6,2),
    headshot_url        TEXT,
    created_at          TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hpg_position_score ON historical_prospect_grades(position, prospect_score);
CREATE INDEX IF NOT EXISTS idx_hpg_year           ON historical_prospect_grades(draft_class_year);
CREATE INDEX IF NOT EXISTS idx_hpg_sleeper        ON historical_prospect_grades(sleeper_id) WHERE sleeper_id IS NOT NULL;

-- ── 2018 Draft Class ──────────────────────────────────────────────────────────
INSERT INTO historical_prospect_grades
  (player_id, name, position, draft_class_year, school, prospect_score, tier, tier_label, overall_rank, position_rank, actual_pick, actual_round, actual_nfl_team)
VALUES
  ('HIST_2018_SAQUON_BARKLEY',   'Saquon Barkley',    'RB', 2018, 'Penn State',    94.2, 1, 'Elite Prospect',   1,  1,  2, 1, 'NYG'),
  ('HIST_2018_JOSH_ALLEN',       'Josh Allen',         'QB', 2018, 'Wyoming',       81.3, 2, 'Strong Prospect',  7,  7,  7, 1, 'BUF'),
  ('HIST_2018_BAKER_MAYFIELD',   'Baker Mayfield',     'QB', 2018, 'Oklahoma',      77.4, 2, 'Strong Prospect',  1,  1,  1, 1, 'CLE'),
  ('HIST_2018_LAMAR_JACKSON',    'Lamar Jackson',      'QB', 2018, 'Louisville',    85.6, 2, 'Strong Prospect',  6,  6, 32, 1, 'BAL'),
  ('HIST_2018_SAM_DARNOLD',      'Sam Darnold',        'QB', 2018, 'USC',           74.8, 3, 'Solid Prospect',   3,  3,  3, 1, 'NYJ'),
  ('HIST_2018_NICK_CHUBB',       'Nick Chubb',         'RB', 2018, 'Georgia',       80.7, 2, 'Strong Prospect',  4,  2, 35, 2, 'CLE'),
  ('HIST_2018_SONY_MICHEL',      'Sony Michel',        'RB', 2018, 'Georgia',       70.4, 3, 'Solid Prospect',  10,  3, 31, 1, 'NE'),
  ('HIST_2018_KERRYON_JOHNSON',  'Kerryon Johnson',    'RB', 2018, 'Auburn',        67.9, 3, 'Solid Prospect',  14,  4, 43, 2, 'DET'),
  ('HIST_2018_CALVIN_RIDLEY',    'Calvin Ridley',      'WR', 2018, 'Alabama',       80.2, 2, 'Strong Prospect',  5,  1, 26, 1, 'ATL'),
  ('HIST_2018_DJ_MOORE',         'D.J. Moore',         'WR', 2018, 'Maryland',      74.6, 3, 'Solid Prospect',   9,  2, 24, 1, 'CAR'),
  ('HIST_2018_COURTLAND_SUTTON', 'Courtland Sutton',   'WR', 2018, 'SMU',           71.3, 3, 'Solid Prospect',  12,  3, 40, 2, 'DEN'),
  ('HIST_2018_MIKE_GESICKI',     'Mike Gesicki',       'TE', 2018, 'Penn State',    71.9, 3, 'Solid Prospect',   2,  1, 42, 2, 'MIA')
ON CONFLICT (player_id) DO NOTHING;

-- ── 2019 Draft Class ──────────────────────────────────────────────────────────
INSERT INTO historical_prospect_grades
  (player_id, name, position, draft_class_year, school, prospect_score, tier, tier_label, overall_rank, position_rank, actual_pick, actual_round, actual_nfl_team)
VALUES
  ('HIST_2019_KYLER_MURRAY',      'Kyler Murray',      'QB', 2019, 'Oklahoma',      87.4, 2, 'Strong Prospect',  1,  1,  1, 1, 'ARI'),
  ('HIST_2019_DWAYNE_HASKINS',    'Dwayne Haskins',    'QB', 2019, 'Ohio State',    72.8, 3, 'Solid Prospect',   4,  2, 15, 1, 'WAS'),
  ('HIST_2019_DANIEL_JONES',      'Daniel Jones',      'QB', 2019, 'Duke',          65.3, 3, 'Solid Prospect',   6,  3,  6, 1, 'NYG'),
  ('HIST_2019_DREW_LOCK',         'Drew Lock',         'QB', 2019, 'Missouri',      61.2, 4, 'Developmental',    8,  4, 42, 2, 'DEN'),
  ('HIST_2019_JOSH_JACOBS',       'Josh Jacobs',       'RB', 2019, 'Alabama',       78.4, 2, 'Strong Prospect',  2,  1, 24, 1, 'LV'),
  ('HIST_2019_MILES_SANDERS',     'Miles Sanders',     'RB', 2019, 'Penn State',    72.1, 3, 'Solid Prospect',   5,  2, 53, 2, 'PHI'),
  ('HIST_2019_DAVID_MONTGOMERY',  'David Montgomery',  'RB', 2019, 'Iowa State',    68.7, 3, 'Solid Prospect',   7,  3, 73, 3, 'CHI'),
  ('HIST_2019_DARRELL_HENDERSON', 'Darrell Henderson', 'RB', 2019, 'Memphis',       65.8, 3, 'Solid Prospect',  10,  5, 70, 3, 'LAR'),
  ('HIST_2019_AJ_BROWN',          'A.J. Brown',        'WR', 2019, 'Ole Miss',      82.6, 2, 'Strong Prospect',  3,  1, 51, 2, 'TEN'),
  ('HIST_2019_DK_METCALF',        'D.K. Metcalf',      'WR', 2019, 'Ole Miss',      79.3, 2, 'Strong Prospect',  4,  2, 64, 2, 'SEA'),
  ('HIST_2019_DEEBO_SAMUEL',      'Deebo Samuel',      'WR', 2019, 'South Carolina',74.2, 3, 'Solid Prospect',   5,  3, 36, 2, 'SF'),
  ('HIST_2019_TERRY_MCLAURIN',    'Terry McLaurin',    'WR', 2019, 'Ohio State',    69.8, 3, 'Solid Prospect',   8,  5, 76, 3, 'WAS'),
  ('HIST_2019_TJ_HOCKENSON',      'T.J. Hockenson',    'TE', 2019, 'Iowa',          83.6, 2, 'Strong Prospect',  1,  1,  8, 1, 'DET'),
  ('HIST_2019_NOAH_FANT',         'Noah Fant',         'TE', 2019, 'Iowa',          76.3, 2, 'Strong Prospect',  2,  2, 20, 1, 'DEN')
ON CONFLICT (player_id) DO NOTHING;

-- ── 2020 Draft Class ──────────────────────────────────────────────────────────
INSERT INTO historical_prospect_grades
  (player_id, name, position, draft_class_year, school, prospect_score, tier, tier_label, overall_rank, position_rank, actual_pick, actual_round, actual_nfl_team)
VALUES
  ('HIST_2020_JOE_BURROW',             'Joe Burrow',              'QB', 2020, 'LSU',          91.6, 1, 'Elite Prospect',  1,  1,  1, 1, 'CIN'),
  ('HIST_2020_TUA_TAGOVAILOA',         'Tua Tagovailoa',          'QB', 2020, 'Alabama',      83.4, 2, 'Strong Prospect', 4,  2,  5, 1, 'MIA'),
  ('HIST_2020_JUSTIN_HERBERT',         'Justin Herbert',          'QB', 2020, 'Oregon',       87.1, 2, 'Strong Prospect', 3,  3,  6, 1, 'LAC'),
  ('HIST_2020_JONATHAN_TAYLOR',        'Jonathan Taylor',         'RB', 2020, 'Wisconsin',    90.2, 1, 'Elite Prospect',  2,  1, 41, 2, 'IND'),
  ('HIST_2020_DANDRE_SWIFT',           "D'Andre Swift",           'RB', 2020, 'Georgia',      84.7, 2, 'Strong Prospect', 3,  2, 35, 2, 'DET'),
  ('HIST_2020_CLYDE_EDWARDS_HELAIRE',  'Clyde Edwards-Helaire',   'RB', 2020, 'LSU',          77.4, 2, 'Strong Prospect', 5,  3, 32, 1, 'KC'),
  ('HIST_2020_CAM_AKERS',              'Cam Akers',               'RB', 2020, 'Florida State',75.8, 3, 'Solid Prospect',  6,  4, 52, 2, 'LAR'),
  ('HIST_2020_AJ_DILLON',             'A.J. Dillon',             'RB', 2020, 'Boston College',68.9, 3, 'Solid Prospect', 11,  7, 62, 2, 'GB'),
  ('HIST_2020_CEEDEE_LAMB',            'CeeDee Lamb',             'WR', 2020, 'Oklahoma',     90.4, 1, 'Elite Prospect',  1,  1, 17, 1, 'DAL'),
  ('HIST_2020_JUSTIN_JEFFERSON',       'Justin Jefferson',        'WR', 2020, 'LSU',          88.9, 2, 'Strong Prospect', 2,  2, 22, 1, 'MIN'),
  ('HIST_2020_JERRY_JEUDY',            'Jerry Jeudy',             'WR', 2020, 'Alabama',      83.1, 2, 'Strong Prospect', 3,  3, 15, 1, 'DEN'),
  ('HIST_2020_TEE_HIGGINS',            'Tee Higgins',             'WR', 2020, 'Clemson',      78.6, 2, 'Strong Prospect', 5,  4, 33, 2, 'CIN'),
  ('HIST_2020_HENRY_RUGGS',            'Henry Ruggs III',         'WR', 2020, 'Alabama',      73.4, 3, 'Solid Prospect',  7,  5, 12, 1, 'LV'),
  ('HIST_2020_BRANDON_AIYUK',          'Brandon Aiyuk',           'WR', 2020, 'Arizona State',74.8, 3, 'Solid Prospect',  6,  6, 25, 1, 'SF')
ON CONFLICT (player_id) DO NOTHING;

-- ── 2021 Draft Class ──────────────────────────────────────────────────────────
INSERT INTO historical_prospect_grades
  (player_id, name, position, draft_class_year, school, prospect_score, tier, tier_label, overall_rank, position_rank, actual_pick, actual_round, actual_nfl_team)
VALUES
  ('HIST_2021_TREVOR_LAWRENCE',  'Trevor Lawrence',  'QB', 2021, 'Clemson',       97.2, 1, 'Elite Prospect',   1,  1,  1, 1, 'JAX'),
  ('HIST_2021_JUSTIN_FIELDS',    'Justin Fields',    'QB', 2021, 'Ohio State',    84.3, 2, 'Strong Prospect',  3,  2, 11, 1, 'CHI'),
  ('HIST_2021_ZACH_WILSON',      'Zach Wilson',      'QB', 2021, 'BYU',           74.6, 3, 'Solid Prospect',   5,  3,  2, 1, 'NYJ'),
  ('HIST_2021_TREY_LANCE',       'Trey Lance',       'QB', 2021, 'NDSU',          72.1, 3, 'Solid Prospect',   6,  4,  3, 1, 'SF'),
  ('HIST_2021_NAJEE_HARRIS',     'Najee Harris',     'RB', 2021, 'Alabama',       76.8, 2, 'Strong Prospect',  4,  1, 24, 1, 'PIT'),
  ('HIST_2021_TRAVIS_ETIENNE',   'Travis Etienne',   'RB', 2021, 'Clemson',       85.4, 2, 'Strong Prospect',  3,  2, 25, 1, 'JAX'),
  ('HIST_2021_JAVONTE_WILLIAMS', 'Javonte Williams', 'RB', 2021, 'North Carolina',78.6, 2, 'Strong Prospect',  5,  3, 35, 2, 'DEN'),
  ('HIST_2021_MICHAEL_CARTER',   'Michael Carter',   'RB', 2021, 'North Carolina',64.2, 4, 'Developmental',   15,  7,107, 4, 'NYJ'),
  ('HIST_2021_JAMARR_CHASE',    "Ja'Marr Chase",    'WR', 2021, 'LSU',           95.6, 1, 'Elite Prospect',   1,  1,  5, 1, 'CIN'),
  ('HIST_2021_JAYLEN_WADDLE',    'Jaylen Waddle',    'WR', 2021, 'Alabama',       86.8, 2, 'Strong Prospect',  2,  2,  6, 1, 'MIA'),
  ('HIST_2021_DEVONTA_SMITH',    "DeVonta Smith",    'WR', 2021, 'Alabama',       84.6, 2, 'Strong Prospect',  3,  3, 10, 1, 'PHI'),
  ('HIST_2021_RASHOD_BATEMAN',   'Rashod Bateman',   'WR', 2021, 'Minnesota',     73.2, 3, 'Solid Prospect',   8,  6, 27, 1, 'BAL'),
  ('HIST_2021_KYLE_PITTS',       'Kyle Pitts',       'TE', 2021, 'Florida',       96.1, 1, 'Elite Prospect',   1,  1,  4, 1, 'ATL'),
  ('HIST_2021_PAT_FREIERMUTH',   'Pat Freiermuth',   'TE', 2021, 'Penn State',    73.8, 3, 'Solid Prospect',   3,  2, 55, 2, 'PIT')
ON CONFLICT (player_id) DO NOTHING;

-- ── 2022 Draft Class ──────────────────────────────────────────────────────────
INSERT INTO historical_prospect_grades
  (player_id, name, position, draft_class_year, school, prospect_score, tier, tier_label, overall_rank, position_rank, actual_pick, actual_round, actual_nfl_team)
VALUES
  ('HIST_2022_KENNY_PICKETT',      'Kenny Pickett',      'QB', 2022, 'Pittsburgh',    68.4, 3, 'Solid Prospect',  6,  1, 20, 1, 'PIT'),
  ('HIST_2022_DESMOND_RIDDER',     'Desmond Ridder',     'QB', 2022, 'Cincinnati',    61.6, 4, 'Developmental',  10,  3, 74, 3, 'ATL'),
  ('HIST_2022_BREECE_HALL',        'Breece Hall',        'RB', 2022, 'Iowa State',    91.4, 1, 'Elite Prospect',  1,  1, 36, 2, 'NYJ'),
  ('HIST_2022_DAMEON_PIERCE',      'Dameon Pierce',      'RB', 2022, 'Florida',       68.3, 3, 'Solid Prospect',  7,  2,107, 4, 'HOU'),
  ('HIST_2022_ISAIAH_SPILLER',     'Isaiah Spiller',     'RB', 2022, 'Texas A&M',     64.7, 4, 'Developmental', 10,  4,123, 4, 'LAC'),
  ('HIST_2022_RACHAAD_WHITE',      'Rachaad White',      'RB', 2022, 'Arizona State', 67.2, 3, 'Solid Prospect',  9,  3, 91, 3, 'TB'),
  ('HIST_2022_GARRETT_WILSON',     'Garrett Wilson',     'WR', 2022, 'Ohio State',    87.3, 2, 'Strong Prospect', 1,  1, 10, 1, 'NYJ'),
  ('HIST_2022_CHRIS_OLAVE',        'Chris Olave',        'WR', 2022, 'Ohio State',    81.8, 2, 'Strong Prospect', 2,  2, 11, 1, 'NO'),
  ('HIST_2022_DRAKE_LONDON',       'Drake London',       'WR', 2022, 'USC',           80.6, 2, 'Strong Prospect', 3,  3,  8, 1, 'ATL'),
  ('HIST_2022_TREYLON_BURKS',      'Treylon Burks',      'WR', 2022, 'Arkansas',      75.8, 3, 'Solid Prospect',  5,  4, 18, 1, 'TEN'),
  ('HIST_2022_GEORGE_PICKENS',     'George Pickens',     'WR', 2022, 'Georgia',       76.4, 2, 'Strong Prospect', 4,  5, 52, 2, 'PIT'),
  ('HIST_2022_CHRISTIAN_WATSON',   'Christian Watson',   'WR', 2022, 'NDSU',          72.3, 3, 'Solid Prospect',  8,  6, 34, 2, 'GB'),
  ('HIST_2022_JAHAN_DOTSON',       'Jahan Dotson',       'WR', 2022, 'Penn State',    73.6, 3, 'Solid Prospect',  6,  7, 16, 1, 'WAS'),
  ('HIST_2022_TREY_MCBRIDE',       'Trey McBride',       'TE', 2022, 'Colorado State',71.4, 3, 'Solid Prospect',  1,  1, 35, 2, 'ARI'),
  ('HIST_2022_GREG_DULCICH',       'Greg Dulcich',       'TE', 2022, 'UCLA',          64.8, 4, 'Developmental',   4,  3, 80, 3, 'DEN')
ON CONFLICT (player_id) DO NOTHING;

-- ── 2023 Draft Class ──────────────────────────────────────────────────────────
INSERT INTO historical_prospect_grades
  (player_id, name, position, draft_class_year, school, prospect_score, tier, tier_label, overall_rank, position_rank, actual_pick, actual_round, actual_nfl_team)
VALUES
  ('HIST_2023_BRYCE_YOUNG',           'Bryce Young',            'QB', 2023, 'Alabama',       89.7, 2, 'Strong Prospect',  1,  1,  1, 1, 'CAR'),
  ('HIST_2023_CJ_STROUD',             'C.J. Stroud',            'QB', 2023, 'Ohio State',    88.4, 2, 'Strong Prospect',  2,  2,  2, 1, 'HOU'),
  ('HIST_2023_ANTHONY_RICHARDSON',    'Anthony Richardson',     'QB', 2023, 'Florida',       83.2, 2, 'Strong Prospect',  4,  3,  4, 1, 'IND'),
  ('HIST_2023_WILL_LEVIS',            'Will Levis',             'QB', 2023, 'Kentucky',      73.4, 3, 'Solid Prospect',   7,  4, 33, 2, 'TEN'),
  ('HIST_2023_BIJAN_ROBINSON',        'Bijan Robinson',         'RB', 2023, 'Texas',         94.8, 1, 'Elite Prospect',   1,  1,  8, 1, 'ATL'),
  ('HIST_2023_ZACH_CHARBONNET',       'Zach Charbonnet',        'RB', 2023, 'UCLA',          77.6, 2, 'Strong Prospect',  3,  2, 52, 2, 'SEA'),
  ('HIST_2023_ROSCHON_JOHNSON',       'Roschon Johnson',        'RB', 2023, 'Texas',         64.3, 4, 'Developmental',   12,  5,115, 4, 'CHI'),
  ('HIST_2023_TANK_BIGSBY',           'Tank Bigsby',            'RB', 2023, 'Auburn',        71.8, 3, 'Solid Prospect',   5,  3, 88, 3, 'JAX'),
  ('HIST_2023_JAXON_SMITH_NJIGBA',    'Jaxon Smith-Njigba',     'WR', 2023, 'Ohio State',    85.9, 2, 'Strong Prospect',  1,  1, 20, 1, 'SEA'),
  ('HIST_2023_JORDAN_ADDISON',        'Jordan Addison',         'WR', 2023, 'USC',           81.4, 2, 'Strong Prospect',  2,  2, 23, 1, 'MIN'),
  ('HIST_2023_ZAY_FLOWERS',           'Zay Flowers',            'WR', 2023, 'Boston College',78.6, 2, 'Strong Prospect',  3,  3, 22, 1, 'BAL'),
  ('HIST_2023_QUENTIN_JOHNSTON',      'Quentin Johnston',       'WR', 2023, 'TCU',           75.2, 3, 'Solid Prospect',   5,  4, 21, 1, 'LAC'),
  ('HIST_2023_RASHEE_RICE',           'Rashee Rice',            'WR', 2023, 'SMU',           74.3, 3, 'Solid Prospect',   6,  5, 55, 2, 'KC'),
  ('HIST_2023_JAYLIN_HYATT',          'Jaylin Hyatt',           'WR', 2023, 'Tennessee',     68.7, 3, 'Solid Prospect',  10,  7, 71, 3, 'NYG'),
  ('HIST_2023_SAM_LAPORTA',           'Sam LaPorta',            'TE', 2023, 'Iowa',          78.4, 2, 'Strong Prospect',  1,  1, 34, 2, 'DET'),
  ('HIST_2023_MICHAEL_MAYER',         'Michael Mayer',          'TE', 2023, 'Notre Dame',    71.8, 3, 'Solid Prospect',   2,  2, 35, 2, 'LV')
ON CONFLICT (player_id) DO NOTHING;

-- ── 2024 Draft Class ──────────────────────────────────────────────────────────
INSERT INTO historical_prospect_grades
  (player_id, name, position, draft_class_year, school, prospect_score, tier, tier_label, overall_rank, position_rank, actual_pick, actual_round, actual_nfl_team)
VALUES
  ('HIST_2024_CALEB_WILLIAMS',     'Caleb Williams',      'QB', 2024, 'USC',             96.8, 1, 'Elite Prospect',   1,  1,  1, 1, 'CHI'),
  ('HIST_2024_DRAKE_MAYE',         'Drake Maye',          'QB', 2024, 'North Carolina',  91.4, 1, 'Elite Prospect',   2,  2,  3, 1, 'NE'),
  ('HIST_2024_JAYDEN_DANIELS',     'Jayden Daniels',      'QB', 2024, 'LSU',             87.6, 2, 'Strong Prospect',  3,  3,  2, 1, 'WAS'),
  ('HIST_2024_JJ_MCCARTHY',        'J.J. McCarthy',       'QB', 2024, 'Michigan',        79.2, 2, 'Strong Prospect',  5,  4, 10, 1, 'MIN'),
  ('HIST_2024_BO_NIX',             'Bo Nix',              'QB', 2024, 'Oregon',          73.4, 3, 'Solid Prospect',   8,  5, 12, 1, 'DEN'),
  ('HIST_2024_MARVIN_HARRISON_JR', 'Marvin Harrison Jr',  'WR', 2024, 'Ohio State',      96.2, 1, 'Elite Prospect',   1,  1,  4, 1, 'ARI'),
  ('HIST_2024_MALIK_NABERS',       'Malik Nabers',        'WR', 2024, 'LSU',             93.4, 1, 'Elite Prospect',   2,  2,  6, 1, 'NYG'),
  ('HIST_2024_ROME_ODUNZE',        'Rome Odunze',         'WR', 2024, 'Washington',      85.8, 2, 'Strong Prospect',  3,  3,  9, 1, 'CHI'),
  ('HIST_2024_BRIAN_THOMAS_JR',    'Brian Thomas Jr',     'WR', 2024, 'LSU',             82.4, 2, 'Strong Prospect',  4,  4, 23, 1, 'JAX'),
  ('HIST_2024_LADD_MCCONKEY',      'Ladd McConkey',       'WR', 2024, 'Georgia',         76.3, 2, 'Strong Prospect',  7,  5, 34, 2, 'LAC'),
  ('HIST_2024_XAVIER_WORTHY',      'Xavier Worthy',       'WR', 2024, 'Texas',           74.8, 3, 'Solid Prospect',   9,  6, 28, 1, 'KC'),
  ('HIST_2024_KEON_COLEMAN',       'Keon Coleman',        'WR', 2024, 'Florida State',   72.6, 3, 'Solid Prospect',  11,  7, 33, 2, 'BUF'),
  ('HIST_2024_ADONAI_MITCHELL',    'Adonai Mitchell',     'WR', 2024, 'Texas',           73.9, 3, 'Solid Prospect',  10,  8, 52, 2, 'IND'),
  ('HIST_2024_RICKY_PEARSALL',     'Ricky Pearsall',      'WR', 2024, 'Florida',         72.1, 3, 'Solid Prospect',  12,  9, 31, 1, 'SF'),
  ('HIST_2024_JONATHON_BROOKS',    'Jonathon Brooks',     'RB', 2024, 'Texas',           81.8, 2, 'Strong Prospect',  2,  1, 46, 2, 'CAR'),
  ('HIST_2024_BUCKY_IRVING',       'Bucky Irving',        'RB', 2024, 'Oregon',          75.6, 3, 'Solid Prospect',  4,  2,125, 4, 'TB'),
  ('HIST_2024_TREY_BENSON',        'Trey Benson',         'RB', 2024, 'Florida State',   72.4, 3, 'Solid Prospect',  5,  3, 93, 3, 'ARI'),
  ('HIST_2024_BLAKE_CORUM',        'Blake Corum',         'RB', 2024, 'Michigan',        67.2, 3, 'Solid Prospect',  8,  5, 86, 3, 'LAR'),
  ('HIST_2024_BROCK_BOWERS',       'Brock Bowers',        'TE', 2024, 'Georgia',         96.4, 1, 'Elite Prospect',  1,  1, 13, 1, 'LV'),
  ('HIST_2024_BEN_SINNOTT',        'Ben Sinnott',         'TE', 2024, 'Kansas State',    67.8, 3, 'Solid Prospect',  4,  3, 61, 2, 'WAS')
ON CONFLICT (player_id) DO NOTHING;

-- ── 2025 Draft Class (drafted April 2025, now historical) ─────────────────────
INSERT INTO historical_prospect_grades
  (player_id, name, position, draft_class_year, school, prospect_score, tier, tier_label, overall_rank, position_rank, actual_pick, actual_round, actual_nfl_team)
VALUES
  ('HIST_2025_CAM_WARD',           'Cam Ward',            'QB', 2025, 'Miami',          89.4, 2, 'Strong Prospect',  1,  1,  1, 1, 'TEN'),
  ('HIST_2025_SHEDEUR_SANDERS',    'Shedeur Sanders',     'QB', 2025, 'Colorado',       84.2, 2, 'Strong Prospect',  3,  2,  5, 1, 'CLE'),
  ('HIST_2025_DILLON_GABRIEL',     'Dillon Gabriel',      'QB', 2025, 'Oregon',         70.6, 3, 'Solid Prospect',  10,  5, 94, 3, 'LAR'),
  ('HIST_2025_JALEN_MILROE',       'Jalen Milroe',        'QB', 2025, 'Alabama',        72.4, 3, 'Solid Prospect',   8,  4, 76, 3, 'SEA'),
  ('HIST_2025_ASHTON_JEANTY',      'Ashton Jeanty',       'RB', 2025, 'Boise State',    93.8, 1, 'Elite Prospect',   1,  1,  9, 1, 'LV'),
  ('HIST_2025_OMARION_HAMPTON',    'Omarion Hampton',     'RB', 2025, 'North Carolina', 85.6, 2, 'Strong Prospect',  2,  2, 22, 1, 'LAC'),
  ('HIST_2025_QUINSHON_JUDKINS',   'Quinshon Judkins',    'RB', 2025, 'Ohio State',     78.3, 2, 'Strong Prospect',  4,  3, 37, 2, 'NE'),
  ('HIST_2025_TREVEYON_HENDERSON', 'TreVeyon Henderson',  'RB', 2025, 'Ohio State',     74.8, 3, 'Solid Prospect',   5,  4, 40, 2, 'NE'),
  ('HIST_2025_DYLAN_SAMPSON',      'Dylan Sampson',       'RB', 2025, 'Tennessee',      67.3, 3, 'Solid Prospect',   9,  6,119, 4, 'PIT'),
  ('HIST_2025_TETAIROA_MCMILLAN',  'Tetairoa McMillan',   'WR', 2025, 'Arizona',        90.8, 1, 'Elite Prospect',   1,  1,  8, 1, 'CAR'),
  ('HIST_2025_TRAVIS_HUNTER',      'Travis Hunter',       'WR', 2025, 'Colorado',       91.6, 1, 'Elite Prospect',   2,  2,  2, 1, 'JAX'),
  ('HIST_2025_LUTHER_BURDEN_III',  'Luther Burden III',   'WR', 2025, 'Missouri',       86.4, 2, 'Strong Prospect',  3,  3, 47, 2, 'CHI'),
  ('HIST_2025_EMEKA_EGBUKA',       'Emeka Egbuka',        'WR', 2025, 'Ohio State',     81.7, 2, 'Strong Prospect',  4,  4, 33, 1, 'TB'),
  ('HIST_2025_TRE_HARRIS',         'Tre Harris',          'WR', 2025, 'Mississippi',    78.9, 2, 'Strong Prospect',  5,  5, 45, 2, 'LAC'),
  ('HIST_2025_MATTHEW_GOLDEN',     'Matthew Golden',      'WR', 2025, 'Texas',          77.4, 2, 'Strong Prospect',  6,  6, 23, 1, 'GB'),
  ('HIST_2025_COLSTON_LOVELAND',   'Colston Loveland',    'TE', 2025, 'Michigan',       87.6, 2, 'Strong Prospect',  1,  1, 10, 1, 'CHI'),
  ('HIST_2025_TYLER_WARREN',       'Tyler Warren',        'TE', 2025, 'Penn State',     85.2, 2, 'Strong Prospect',  2,  2, 16, 1, 'IND'),
  ('HIST_2025_HAROLD_FANNIN_JR',   'Harold Fannin Jr',    'TE', 2025, 'Bowling Green',  79.6, 2, 'Strong Prospect',  3,  3, 49, 2, 'CLE'),
  ('HIST_2025_MASON_TAYLOR',       'Mason Taylor',        'TE', 2025, 'LSU',            74.3, 3, 'Solid Prospect',   4,  4, 42, 2, 'NYJ')
ON CONFLICT (player_id) DO NOTHING;
