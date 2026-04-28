-- Historical prospect grades for draft classes 2018-2025.
-- Identity/draft data is seeded here. Prospect scores (prospect_score, tier,
-- component scores, etc.) are populated by running:
--   python scripts/populate_historical_grades.py
-- which executes the real prospect model via the backtest pipeline.

create TABLE IF NOT EXISTS historical_prospect_grades (
    player_id           TEXT        PRIMARY KEY,  -- HIST_{YEAR}_{NAME_SLUG}
    sleeper_id          TEXT,
    name                TEXT        NOT NULL,
    position            TEXT        NOT NULL,
    draft_class_year    INTEGER     NOT NULL,
    school              TEXT,
    -- Scores populated by backtest pipeline (NULL until script is run)
    prospect_score      DECIMAL(6,2),
    tier                INTEGER,
    tier_label          TEXT,
    overall_rank        INTEGER,
    position_rank       INTEGER,
    production_score    DECIMAL(6,2),
    efficiency_score    DECIMAL(6,2),
    age_score           DECIMAL(6,2),
    breakout_profile_score DECIMAL(6,2),
    athleticism_score   DECIMAL(6,2),
    competition_score   DECIMAL(6,2),
    draft_capital_score DECIMAL(6,2),
    confidence_score    DECIMAL(6,2),
    -- Actual NFL draft results
    actual_pick         INTEGER,
    actual_round        INTEGER,
    actual_nfl_team     TEXT,
    headshot_url        TEXT,
    created_at          TIMESTAMP   DEFAULT NOW()
);

create index IF NOT EXISTS idx_hpg_position_score ON historical_prospect_grades(position, prospect_score);
create index IF NOT EXISTS idx_hpg_year           ON historical_prospect_grades(draft_class_year);
create index IF NOT EXISTS idx_hpg_sleeper        ON historical_prospect_grades(sleeper_id) WHERE sleeper_id IS NOT NULL;

-- ── Identity seed data (no scores — run populate_historical_grades.py for those) ──

-- 2016 + 2017 Top Tier Historical Prospects
insert into historical_prospect_grades (player_id, name, position, draft_class_year, school, actual_pick, actual_round, actual_nfl_team)
values
  ('HIST_2016_JARED_GOFF',        'Jared Goff',          'QB', 2016, 'California',       1, 1, 'LAR'),
  ('HIST_2016_CARSON_WENTZ',      'Carson Wentz',        'QB', 2016, 'North Dakota St',  2, 1, 'PHI'),
  ('HIST_2016_EZEKIEL_ELLIOTT',   'Ezekiel Elliott',     'RB', 2016, 'Ohio State',       4, 1, 'DAL'),
  ('HIST_2016_COREY_COLEMAN',     'Corey Coleman',       'WR', 2016, 'Baylor',          15, 1, 'CLE'),
  ('HIST_2016_WILL_FULLER',       'Will Fuller',         'WR', 2016, 'Notre Dame',      21, 1, 'HOU'),
  ('HIST_2016_JOSH_DOCTSON',      'Josh Doctson',        'WR', 2016, 'TCU',             22, 1, 'WAS'),
  ('HIST_2016_LAQUON_TREADWELL',  'Laquon Treadwell',    'WR', 2016, 'Ole Miss',        23, 1, 'MIN'),
  ('HIST_2016_HUNTER_HENRY',      'Hunter Henry',        'TE', 2016, 'Arkansas',        35, 2, 'SD'),
  ('HIST_2016_DERRICK_HENRY',     'Derrick Henry',       'RB', 2016, 'Alabama',         45, 2, 'TEN'),
  ('HIST_2016_MICHAEL_THOMAS',    'Michael Thomas',      'WR', 2016, 'Ohio State',      47, 2, 'NO'),
  ('HIST_2016_TYLER_BOYD',        'Tyler Boyd',          'WR', 2016, 'Pittsburgh',      55, 2, 'CIN'),

  -- 2017
  ('HIST_2017_MITCHELL_TRUBISKY', 'Mitchell Trubisky',   'QB', 2017, 'North Carolina',   2, 1, 'CHI'),
  ('HIST_2017_LEONARD_FOURNETTE', 'Leonard Fournette',   'RB', 2017, 'LSU',              4, 1, 'JAX'),
  ('HIST_2017_COREY_DAVIS',       'Corey Davis',         'WR', 2017, 'Western Michigan', 5, 1, 'TEN'),
  ('HIST_2017_MIKE_WILLIAMS',     'Mike Williams',       'WR', 2017, 'Clemson',          7, 1, 'LAC'),
  ('HIST_2017_CHRISTIAN_MCCAFFREY','Christian McCaffrey','RB', 2017, 'Stanford',         8, 1, 'CAR'),
  ('HIST_2017_JOHN_ROSS',         'John Ross',           'WR', 2017, 'Washington',       9, 1, 'CIN'),
  ('HIST_2017_PATRICK_MAHOMES',   'Patrick Mahomes',     'QB', 2017, 'Texas Tech',      10, 1, 'KC'),
  ('HIST_2017_DESHAUN_WATSON',    'Deshaun Watson',      'QB', 2017, 'Clemson',         12, 1, 'HOU'),
  ('HIST_2017_OJ_HOWARD',         'O.J. Howard',         'TE', 2017, 'Alabama',         19, 1, 'TB'),
  ('HIST_2017_EVAN_ENGRAM',       'Evan Engram',         'TE', 2017, 'Ole Miss',        23, 1, 'NYG'),
  ('HIST_2017_DAVID_NJOKU',       'David Njoku',         'TE', 2017, 'Miami',           29, 1, 'CLE'),
  ('HIST_2017_DALVIN_COOK',       'Dalvin Cook',         'RB', 2017, 'Florida State',   41, 2, 'MIN'),
  ('HIST_2017_JOE_MIXON',         'Joe Mixon',           'RB', 2017, 'Oklahoma',        48, 2, 'CIN'),
  ('HIST_2017_JUJU_SMITH_SCHUSTER','JuJu Smith-Schuster','WR', 2017, 'USC',             62, 2, 'PIT'),
  ('HIST_2017_ALVIN_KAMARA',      'Alvin Kamara',        'RB', 2017, 'Tennessee',       67, 3, 'NO'),
  ('HIST_2017_COOPER_KUPP',       'Cooper Kupp',         'WR', 2017, 'Eastern Washington',69,3,'LAR')
ON CONFLICT (player_id) DO NOTHING;

-- 2018 Draft Class
insert into historical_prospect_grades (player_id, name, position, draft_class_year, school, actual_pick, actual_round, actual_nfl_team)
values
  ('HIST_2018_SAQUON_BARKLEY',   'Saquon Barkley',    'RB', 2018, 'Penn State',     2, 1, 'NYG'),
  ('HIST_2018_JOSH_ALLEN',       'Josh Allen',         'QB', 2018, 'Wyoming',        7, 1, 'BUF'),
  ('HIST_2018_BAKER_MAYFIELD',   'Baker Mayfield',     'QB', 2018, 'Oklahoma',       1, 1, 'CLE'),
  ('HIST_2018_LAMAR_JACKSON',    'Lamar Jackson',      'QB', 2018, 'Louisville',    32, 1, 'BAL'),
  ('HIST_2018_SAM_DARNOLD',      'Sam Darnold',        'QB', 2018, 'USC',            3, 1, 'NYJ'),
  ('HIST_2018_NICK_CHUBB',       'Nick Chubb',         'RB', 2018, 'Georgia',       35, 2, 'CLE'),
  ('HIST_2018_SONY_MICHEL',      'Sony Michel',        'RB', 2018, 'Georgia',       31, 1, 'NE'),
  ('HIST_2018_KERRYON_JOHNSON',  'Kerryon Johnson',    'RB', 2018, 'Auburn',        43, 2, 'DET'),
  ('HIST_2018_CALVIN_RIDLEY',    'Calvin Ridley',      'WR', 2018, 'Alabama',       26, 1, 'ATL'),
  ('HIST_2018_DJ_MOORE',         'D.J. Moore',         'WR', 2018, 'Maryland',      24, 1, 'CAR'),
  ('HIST_2018_COURTLAND_SUTTON', 'Courtland Sutton',   'WR', 2018, 'SMU',           40, 2, 'DEN'),
  ('HIST_2018_MIKE_GESICKI',     'Mike Gesicki',       'TE', 2018, 'Penn State',    42, 2, 'MIA')
ON CONFLICT (player_id) DO NOTHING;

-- 2019 Draft Class
insert into historical_prospect_grades (player_id, name, position, draft_class_year, school, actual_pick, actual_round, actual_nfl_team)
values
  ('HIST_2019_KYLER_MURRAY',      'Kyler Murray',      'QB', 2019, 'Oklahoma',       1, 1, 'ARI'),
  ('HIST_2019_DWAYNE_HASKINS',    'Dwayne Haskins',    'QB', 2019, 'Ohio State',    15, 1, 'WAS'),
  ('HIST_2019_DANIEL_JONES',      'Daniel Jones',      'QB', 2019, 'Duke',           6, 1, 'NYG'),
  ('HIST_2019_DREW_LOCK',         'Drew Lock',         'QB', 2019, 'Missouri',      42, 2, 'DEN'),
  ('HIST_2019_JOSH_JACOBS',       'Josh Jacobs',       'RB', 2019, 'Alabama',       24, 1, 'LV'),
  ('HIST_2019_MILES_SANDERS',     'Miles Sanders',     'RB', 2019, 'Penn State',    53, 2, 'PHI'),
  ('HIST_2019_DAVID_MONTGOMERY',  'David Montgomery',  'RB', 2019, 'Iowa State',    73, 3, 'CHI'),
  ('HIST_2019_DARRELL_HENDERSON', 'Darrell Henderson', 'RB', 2019, 'Memphis',       70, 3, 'LAR'),
  ('HIST_2019_AJ_BROWN',          'A.J. Brown',        'WR', 2019, 'Ole Miss',      51, 2, 'TEN'),
  ('HIST_2019_DK_METCALF',        'D.K. Metcalf',      'WR', 2019, 'Ole Miss',      64, 2, 'SEA'),
  ('HIST_2019_DEEBO_SAMUEL',      'Deebo Samuel',      'WR', 2019, 'South Carolina',36, 2, 'SF'),
  ('HIST_2019_TERRY_MCLAURIN',    'Terry McLaurin',    'WR', 2019, 'Ohio State',    76, 3, 'WAS'),
  ('HIST_2019_TJ_HOCKENSON',      'T.J. Hockenson',    'TE', 2019, 'Iowa',           8, 1, 'DET'),
  ('HIST_2019_NOAH_FANT',         'Noah Fant',         'TE', 2019, 'Iowa',          20, 1, 'DEN')
ON CONFLICT (player_id) DO NOTHING;

-- 2020 Draft Class
insert into historical_prospect_grades (player_id, name, position, draft_class_year, school, actual_pick, actual_round, actual_nfl_team)
values
  ('HIST_2020_JOE_BURROW',             'Joe Burrow',              'QB', 2020, 'LSU',           1, 1, 'CIN'),
  ('HIST_2020_TUA_TAGOVAILOA',         'Tua Tagovailoa',          'QB', 2020, 'Alabama',       5, 1, 'MIA'),
  ('HIST_2020_JUSTIN_HERBERT',         'Justin Herbert',          'QB', 2020, 'Oregon',        6, 1, 'LAC'),
  ('HIST_2020_JONATHAN_TAYLOR',        'Jonathan Taylor',         'RB', 2020, 'Wisconsin',    41, 2, 'IND'),
  ('HIST_2020_DANDRE_SWIFT',           'D''Andre Swift',          'RB', 2020, 'Georgia',      35, 2, 'DET'),
  ('HIST_2020_CLYDE_EDWARDS_HELAIRE',  'Clyde Edwards-Helaire',   'RB', 2020, 'LSU',          32, 1, 'KC'),
  ('HIST_2020_CAM_AKERS',              'Cam Akers',               'RB', 2020, 'Florida State',52, 2, 'LAR'),
  ('HIST_2020_AJ_DILLON',             'A.J. Dillon',              'RB', 2020, 'Boston College',62, 2, 'GB'),
  ('HIST_2020_CEEDEE_LAMB',            'CeeDee Lamb',             'WR', 2020, 'Oklahoma',     17, 1, 'DAL'),
  ('HIST_2020_JUSTIN_JEFFERSON',       'Justin Jefferson',        'WR', 2020, 'LSU',          22, 1, 'MIN'),
  ('HIST_2020_JERRY_JEUDY',            'Jerry Jeudy',             'WR', 2020, 'Alabama',      15, 1, 'DEN'),
  ('HIST_2020_TEE_HIGGINS',            'Tee Higgins',             'WR', 2020, 'Clemson',      33, 2, 'CIN'),
  ('HIST_2020_HENRY_RUGGS',            'Henry Ruggs III',         'WR', 2020, 'Alabama',      12, 1, 'LV'),
  ('HIST_2020_BRANDON_AIYUK',          'Brandon Aiyuk',           'WR', 2020, 'Arizona State',25, 1, 'SF')
ON CONFLICT (player_id) DO NOTHING;

-- 2021 Draft Class
insert into historical_prospect_grades (player_id, name, position, draft_class_year, school, actual_pick, actual_round, actual_nfl_team)
values
  ('HIST_2021_TREVOR_LAWRENCE',  'Trevor Lawrence',  'QB', 2021, 'Clemson',        1, 1, 'JAX'),
  ('HIST_2021_JUSTIN_FIELDS',    'Justin Fields',    'QB', 2021, 'Ohio State',    11, 1, 'CHI'),
  ('HIST_2021_ZACH_WILSON',      'Zach Wilson',      'QB', 2021, 'BYU',            2, 1, 'NYJ'),
  ('HIST_2021_TREY_LANCE',       'Trey Lance',       'QB', 2021, 'NDSU',           3, 1, 'SF'),
  ('HIST_2021_NAJEE_HARRIS',     'Najee Harris',     'RB', 2021, 'Alabama',       24, 1, 'PIT'),
  ('HIST_2021_TRAVIS_ETIENNE',   'Travis Etienne',   'RB', 2021, 'Clemson',       25, 1, 'JAX'),
  ('HIST_2021_JAVONTE_WILLIAMS', 'Javonte Williams', 'RB', 2021, 'North Carolina',35, 2, 'DEN'),
  ('HIST_2021_MICHAEL_CARTER',   'Michael Carter',   'RB', 2021, 'North Carolina',107,4, 'NYJ'),
  ('HIST_2021_JAMARR_CHASE',    'Ja''Marr Chase',    'WR', 2021, 'LSU',            5, 1, 'CIN'),
  ('HIST_2021_JAYLEN_WADDLE',    'Jaylen Waddle',    'WR', 2021, 'Alabama',        6, 1, 'MIA'),
  ('HIST_2021_DEVONTA_SMITH',    'DeVonta Smith',    'WR', 2021, 'Alabama',       10, 1, 'PHI'),
  ('HIST_2021_RASHOD_BATEMAN',   'Rashod Bateman',   'WR', 2021, 'Minnesota',     27, 1, 'BAL'),
  ('HIST_2021_KYLE_PITTS',       'Kyle Pitts',       'TE', 2021, 'Florida',        4, 1, 'ATL'),
  ('HIST_2021_PAT_FREIERMUTH',   'Pat Freiermuth',   'TE', 2021, 'Penn State',    55, 2, 'PIT'),
  ('HIST_2021_AMON_RA_ST_BROWN', 'Amon-Ra St. Brown','WR', 2021, 'USC',           112, 4, 'DET'),
  ('HIST_2021_ELIJAH_MOORE',     'Elijah Moore',     'WR', 2021, 'Ole Miss',       34, 2, 'NYJ'),
  ('HIST_2020_DARNELL_MOONEY',   'Darnell Mooney',   'WR', 2020, 'Tulane',        173, 5, 'CHI'),
  ('HIST_2020_GABRIEL_DAVIS',    'Gabriel Davis',    'WR', 2020, 'UCF',           128, 4, 'BUF')
ON CONFLICT (player_id) DO NOTHING;

-- 2022 Draft Class
insert into historical_prospect_grades (player_id, name, position, draft_class_year, school, actual_pick, actual_round, actual_nfl_team)
values
  ('HIST_2022_KENNY_PICKETT',      'Kenny Pickett',      'QB', 2022, 'Pittsburgh',     20, 1, 'PIT'),
  ('HIST_2022_DESMOND_RIDDER',     'Desmond Ridder',     'QB', 2022, 'Cincinnati',     74, 3, 'ATL'),
  ('HIST_2022_BREECE_HALL',        'Breece Hall',        'RB', 2022, 'Iowa State',     36, 2, 'NYJ'),
  ('HIST_2022_DAMEON_PIERCE',      'Dameon Pierce',      'RB', 2022, 'Florida',       107, 4, 'HOU'),
  ('HIST_2022_ISAIAH_SPILLER',     'Isaiah Spiller',     'RB', 2022, 'Texas A&M',     123, 4, 'LAC'),
  ('HIST_2022_RACHAAD_WHITE',      'Rachaad White',      'RB', 2022, 'Arizona State',  91, 3, 'TB'),
  ('HIST_2022_GARRETT_WILSON',     'Garrett Wilson',     'WR', 2022, 'Ohio State',     10, 1, 'NYJ'),
  ('HIST_2022_CHRIS_OLAVE',        'Chris Olave',        'WR', 2022, 'Ohio State',     11, 1, 'NO'),
  ('HIST_2022_DRAKE_LONDON',       'Drake London',       'WR', 2022, 'USC',             8, 1, 'ATL'),
  ('HIST_2022_TREYLON_BURKS',      'Treylon Burks',      'WR', 2022, 'Arkansas',       18, 1, 'TEN'),
  ('HIST_2022_GEORGE_PICKENS',     'George Pickens',     'WR', 2022, 'Georgia',        52, 2, 'PIT'),
  ('HIST_2022_CHRISTIAN_WATSON',   'Christian Watson',   'WR', 2022, 'NDSU',           34, 2, 'GB'),
  ('HIST_2022_JAHAN_DOTSON',       'Jahan Dotson',       'WR', 2022, 'Penn State',     16, 1, 'WAS'),
  ('HIST_2022_TREY_MCBRIDE',       'Trey McBride',       'TE', 2022, 'Colorado State', 35, 2, 'ARI'),
  ('HIST_2022_GREG_DULCICH',       'Greg Dulcich',       'TE', 2022, 'UCLA',           80, 3, 'DEN')
ON CONFLICT (player_id) DO NOTHING;

-- 2023 Draft Class
insert into historical_prospect_grades (player_id, name, position, draft_class_year, school, actual_pick, actual_round, actual_nfl_team)
values
  ('HIST_2023_BRYCE_YOUNG',           'Bryce Young',            'QB', 2023, 'Alabama',        1, 1, 'CAR'),
  ('HIST_2023_CJ_STROUD',             'C.J. Stroud',            'QB', 2023, 'Ohio State',      2, 1, 'HOU'),
  ('HIST_2023_ANTHONY_RICHARDSON',    'Anthony Richardson',     'QB', 2023, 'Florida',         4, 1, 'IND'),
  ('HIST_2023_WILL_LEVIS',            'Will Levis',             'QB', 2023, 'Kentucky',       33, 2, 'TEN'),
  ('HIST_2023_BIJAN_ROBINSON',        'Bijan Robinson',         'RB', 2023, 'Texas',           8, 1, 'ATL'),
  ('HIST_2023_ZACH_CHARBONNET',       'Zach Charbonnet',        'RB', 2023, 'UCLA',           52, 2, 'SEA'),
  ('HIST_2023_ROSCHON_JOHNSON',       'Roschon Johnson',        'RB', 2023, 'Texas',          115, 4, 'CHI'),
  ('HIST_2023_TANK_BIGSBY',           'Tank Bigsby',            'RB', 2023, 'Auburn',          88, 3, 'JAX'),
  ('HIST_2023_JAXON_SMITH_NJIGBA',    'Jaxon Smith-Njigba',     'WR', 2023, 'Ohio State',     20, 1, 'SEA'),
  ('HIST_2023_JORDAN_ADDISON',        'Jordan Addison',         'WR', 2023, 'USC',            23, 1, 'MIN'),
  ('HIST_2023_ZAY_FLOWERS',           'Zay Flowers',            'WR', 2023, 'Boston College', 22, 1, 'BAL'),
  ('HIST_2023_QUENTIN_JOHNSTON',      'Quentin Johnston',       'WR', 2023, 'TCU',            21, 1, 'LAC'),
  ('HIST_2023_RASHEE_RICE',           'Rashee Rice',            'WR', 2023, 'SMU',            55, 2, 'KC'),
  ('HIST_2023_JAYLIN_HYATT',          'Jaylin Hyatt',           'WR', 2023, 'Tennessee',      71, 3, 'NYG'),
  ('HIST_2023_SAM_LAPORTA',           'Sam LaPorta',            'TE', 2023, 'Iowa',           34, 2, 'DET'),
  ('HIST_2023_MICHAEL_MAYER',         'Michael Mayer',          'TE', 2023, 'Notre Dame',     35, 2, 'LV'),
  ('HIST_2023_DEVON_ACHANE',          'DeVon Achane',           'RB', 2023, 'Texas A&M',      84, 3, 'MIA'),
  ('HIST_2023_JOSH_DOWNS',            'Josh Downs',             'WR', 2023, 'North Carolina', 79, 3, 'IND'),
  ('HIST_2023_TYJAE_SPEARS',          'Tyjae Spears',           'RB', 2023, 'Tulane',         81, 3, 'TEN'),
  ('HIST_2023_MARVIN_MIMS',           'Marvin Mims Jr',         'WR', 2023, 'Oklahoma',       63, 2, 'DEN')
ON CONFLICT (player_id) DO NOTHING;

-- 2024 Draft Class
insert into historical_prospect_grades (player_id, name, position, draft_class_year, school, actual_pick, actual_round, actual_nfl_team)
values
  ('HIST_2024_CALEB_WILLIAMS',     'Caleb Williams',      'QB', 2024, 'USC',              1, 1, 'CHI'),
  ('HIST_2024_DRAKE_MAYE',         'Drake Maye',          'QB', 2024, 'North Carolina',   3, 1, 'NE'),
  ('HIST_2024_JAYDEN_DANIELS',     'Jayden Daniels',      'QB', 2024, 'LSU',              2, 1, 'WAS'),
  ('HIST_2024_JJ_MCCARTHY',        'J.J. McCarthy',       'QB', 2024, 'Michigan',        10, 1, 'MIN'),
  ('HIST_2024_BO_NIX',             'Bo Nix',              'QB', 2024, 'Oregon',           12, 1, 'DEN'),
  ('HIST_2024_MARVIN_HARRISON_JR', 'Marvin Harrison Jr',  'WR', 2024, 'Ohio State',       4, 1, 'ARI'),
  ('HIST_2024_MALIK_NABERS',       'Malik Nabers',        'WR', 2024, 'LSU',              6, 1, 'NYG'),
  ('HIST_2024_ROME_ODUNZE',        'Rome Odunze',         'WR', 2024, 'Washington',       9, 1, 'CHI'),
  ('HIST_2024_BRIAN_THOMAS_JR',    'Brian Thomas Jr',     'WR', 2024, 'LSU',             23, 1, 'JAX'),
  ('HIST_2024_LADD_MCCONKEY',      'Ladd McConkey',       'WR', 2024, 'Georgia',         34, 2, 'LAC'),
  ('HIST_2024_XAVIER_WORTHY',      'Xavier Worthy',       'WR', 2024, 'Texas',           28, 1, 'KC'),
  ('HIST_2024_KEON_COLEMAN',       'Keon Coleman',        'WR', 2024, 'Florida State',   33, 2, 'BUF'),
  ('HIST_2024_ADONAI_MITCHELL',    'Adonai Mitchell',     'WR', 2024, 'Texas',           52, 2, 'IND'),
  ('HIST_2024_RICKY_PEARSALL',     'Ricky Pearsall',      'WR', 2024, 'Florida',         31, 1, 'SF'),
  ('HIST_2024_JONATHON_BROOKS',    'Jonathon Brooks',     'RB', 2024, 'Texas',           46, 2, 'CAR'),
  ('HIST_2024_BUCKY_IRVING',       'Bucky Irving',        'RB', 2024, 'Oregon',         125, 4, 'TB'),
  ('HIST_2024_TREY_BENSON',        'Trey Benson',         'RB', 2024, 'Florida State',   93, 3, 'ARI'),
  ('HIST_2024_BLAKE_CORUM',        'Blake Corum',         'RB', 2024, 'Michigan',        86, 3, 'LAR'),
  ('HIST_2024_BROCK_BOWERS',       'Brock Bowers',        'TE', 2024, 'Georgia',         13, 1, 'LV'),
  ('HIST_2024_BEN_SINNOTT',        'Ben Sinnott',         'TE', 2024, 'Kansas State',    61, 2, 'WAS')
ON CONFLICT (player_id) DO NOTHING;

-- 2025 Draft Class (drafted April 2025)
insert into historical_prospect_grades (player_id, name, position, draft_class_year, school, actual_pick, actual_round, actual_nfl_team)
values
  ('HIST_2025_CAM_WARD',           'Cam Ward',            'QB', 2025, 'Miami',           1, 1, 'TEN'),
  ('HIST_2025_SHEDEUR_SANDERS',    'Shedeur Sanders',     'QB', 2025, 'Colorado',        5, 1, 'CLE'),
  ('HIST_2025_DILLON_GABRIEL',     'Dillon Gabriel',      'QB', 2025, 'Oregon',         94, 3, 'LAR'),
  ('HIST_2025_JALEN_MILROE',       'Jalen Milroe',        'QB', 2025, 'Alabama',        76, 3, 'SEA'),
  ('HIST_2025_ASHTON_JEANTY',      'Ashton Jeanty',       'RB', 2025, 'Boise State',    9,  1, 'LV'),
  ('HIST_2025_OMARION_HAMPTON',    'Omarion Hampton',     'RB', 2025, 'North Carolina', 22, 1, 'LAC'),
  ('HIST_2025_QUINSHON_JUDKINS',   'Quinshon Judkins',    'RB', 2025, 'Ohio State',     37, 2, 'NE'),
  ('HIST_2025_TREVEYON_HENDERSON', 'TreVeyon Henderson',  'RB', 2025, 'Ohio State',     40, 2, 'NE'),
  ('HIST_2025_DYLAN_SAMPSON',      'Dylan Sampson',       'RB', 2025, 'Tennessee',     119, 4, 'PIT'),
  ('HIST_2025_TETAIROA_MCMILLAN',  'Tetairoa McMillan',   'WR', 2025, 'Arizona',         8, 1, 'CAR'),
  ('HIST_2025_TRAVIS_HUNTER',      'Travis Hunter',       'WR', 2025, 'Colorado',        2, 1, 'JAX'),
  ('HIST_2025_LUTHER_BURDEN_III',  'Luther Burden III',   'WR', 2025, 'Missouri',       47, 2, 'CHI'),
  ('HIST_2025_EMEKA_EGBUKA',       'Emeka Egbuka',        'WR', 2025, 'Ohio State',     33, 1, 'TB'),
  ('HIST_2025_TRE_HARRIS',         'Tre Harris',          'WR', 2025, 'Mississippi',    45, 2, 'LAC'),
  ('HIST_2025_MATTHEW_GOLDEN',     'Matthew Golden',      'WR', 2025, 'Texas',          23, 1, 'GB'),
  ('HIST_2025_COLSTON_LOVELAND',   'Colston Loveland',    'TE', 2025, 'Michigan',       10, 1, 'CHI'),
  ('HIST_2025_TYLER_WARREN',       'Tyler Warren',        'TE', 2025, 'Penn State',     16, 1, 'IND'),
  ('HIST_2025_HAROLD_FANNIN_JR',   'Harold Fannin Jr',    'TE', 2025, 'Bowling Green',  49, 2, 'CLE'),
  ('HIST_2025_MASON_TAYLOR',       'Mason Taylor',        'TE', 2025, 'LSU',            42, 2, 'NYJ')
ON CONFLICT (player_id) DO NOTHING;
