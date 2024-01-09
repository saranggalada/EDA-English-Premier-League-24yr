#################################################################################

# EPL Viz
# https://epl-viz.streamlit.app/
# Copyright (c) 2024 Sarang Galada

#################################################################################



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


##########################################################################################################


# Gets cumulative match results arranged by teams and matchweek
def cum_results(ds):
    # Create a dictionary with team names as keys
    unique_teams = list(np.sort(ds['HomeTeam'].unique()))
    matchweeks = int(len(ds)/10)

    wins_dict = {}
    draws_dict = {}
    loss_dict = {}
    points_dict = {}

    for i in unique_teams:
        wins_dict[i] = []
        draws_dict[i] = []
        loss_dict[i] = []
        points_dict[i] = []

    # Create new columns for home wins and away wins for each fixture
    ds['HomeWins'] = np.where(ds['FTR'] == 'H', 1, 0)
    ds['AwayWins'] = np.where(ds['FTR'] == 'A', 1, 0)

    # Create new columns for home draws and away draws for each fixture
    ds['HomeDraws'] = np.where(ds['FTR'] == 'D', 1, 0)
    ds['AwayDraws'] = np.where(ds['FTR'] == 'D', 1, 0)

    # Create new columns for home losses and away losses for each fixture
    ds['HomeLosses'] = np.where(ds['FTR'] == 'A', 1, 0)
    ds['AwayLosses'] = np.where(ds['FTR'] == 'H', 1, 0)

    # Create new columns for homepoints and awaypoints for each fixture
    ds['HomePoints'] = np.where(ds['FTR'] == 'H', 3, np.where(ds['FTR'] == 'A', 0, 1))
    ds['AwayPoints'] = np.where(ds['FTR'] == 'A', 3, np.where(ds['FTR'] == 'H', 0, 1))
    
    # the value corresponding to keys is a list containing the match location.
    for i in range(len(ds)):
        HW = ds.iloc[i]['HomeWins']
        AW = ds.iloc[i]['AwayWins']
        HD = ds.iloc[i]['HomeDraws']
        AD = ds.iloc[i]['AwayDraws']
        HL = ds.iloc[i]['HomeLosses']
        AL = ds.iloc[i]['AwayLosses']
        HP = ds.iloc[i]['HomePoints']
        AP = ds.iloc[i]['AwayPoints']

        wins_dict[ds.iloc[i].HomeTeam].append(HW)
        wins_dict[ds.iloc[i].AwayTeam].append(AW)
        draws_dict[ds.iloc[i].HomeTeam].append(HD)
        draws_dict[ds.iloc[i].AwayTeam].append(AD)
        loss_dict[ds.iloc[i].HomeTeam].append(HL)
        loss_dict[ds.iloc[i].AwayTeam].append(AL)
        points_dict[ds.iloc[i].HomeTeam].append(HP)
        points_dict[ds.iloc[i].AwayTeam].append(AP)
    
    # Create a dataframe for league points where rows are teams and cols are matchweek.
    Wins = pd.DataFrame(data=wins_dict, index = [i for i in range(1, matchweeks+1)]).T
    Draws = pd.DataFrame(data=draws_dict, index = [i for i in range(1, matchweeks+1)]).T
    Loss = pd.DataFrame(data=loss_dict, index = [i for i in range(1, matchweeks+1)]).T
    Points = pd.DataFrame(data=points_dict, index = [i for i in range(1, matchweeks+1)]).T
    PrevResult = pd.DataFrame(data=points_dict, index = [i for i in range(1, matchweeks+1)]).T
    Form5M = pd.DataFrame(data=points_dict, index = [i for i in range(1, matchweeks+1)]).T

    # print(Points.head())

    Wins[0] = 0
    Draws[0] = 0
    Loss[0] = 0
    Points[0] = 0
    PrevResult[0] = 0
    Form5M[0] = 0

    # Calculate previous result and 5-match form
    for i in range(2, matchweeks+1):
        PrevResult[i] = Points[i-1]
        if i<6:
            Form5M[i] = 0
            for j in range(1,i):
                Form5M[i] = Form5M[i] + Points[j]
        else:
            Form5M[i] = Points[i-1] + Points[i-2] + Points[i-3] + Points[i-4] + Points[i-5]

    # Aggregate results upto each matchweek
    for i in range(2, matchweeks+1):
        Wins[i] = Wins[i] + Wins[i-1]
        Draws[i] = Draws[i] + Draws[i-1]
        Loss[i] = Loss[i] + Loss[i-1]
        Points[i] = Points[i] + Points[i-1]

    return Wins, Draws, Loss, Points, PrevResult, Form5M



# Gets the cumulative goals scored, conceded and difference arranged by teams and matchweek
def cum_goalstats(ds):

    matchweeks = int(len(ds)/10)
    unique_teams = list(np.sort(ds['HomeTeam'].unique()))

    # Create dictionaries with team names as keys
    gs_dict = {}
    gc_dict = {}
    gd_dict = {}
    sf_dict = {}
    stf_dict = {}
    sc_dict = {}
    stc_dict = {}

    for i in unique_teams:
        gs_dict[i] = []
        gc_dict[i] = []
        gd_dict[i] = []
        sf_dict[i] = []
        stf_dict[i] = []
        sc_dict[i] = []
        stc_dict[i] = []

    for i in range(len(ds)):
        HTGS = ds.iloc[i]['FTHG']
        ATGS = ds.iloc[i]['FTAG']
        HTGC = ds.iloc[i]['FTAG']
        ATGC = ds.iloc[i]['FTHG']
        HTSF = ds.iloc[i]['HS']
        ATSF = ds.iloc[i]['AS']
        HTSTF = ds.iloc[i]['HST']
        ATSTF = ds.iloc[i]['AST']
        HTSC = ds.iloc[i]['AS']
        ATSC = ds.iloc[i]['HS']
        HTSTC = ds.iloc[i]['AST']
        ATSTC = ds.iloc[i]['HST']

        gs_dict[ds.iloc[i].HomeTeam].append(HTGS)
        gs_dict[ds.iloc[i].AwayTeam].append(ATGS)
        gc_dict[ds.iloc[i].HomeTeam].append(HTGC)
        gc_dict[ds.iloc[i].AwayTeam].append(ATGC)
        gd_dict[ds.iloc[i].HomeTeam].append(HTGS - HTGC)
        gd_dict[ds.iloc[i].AwayTeam].append(ATGS - ATGC)
        sf_dict[ds.iloc[i].HomeTeam].append(HTSF)
        sf_dict[ds.iloc[i].AwayTeam].append(ATSF)
        stf_dict[ds.iloc[i].HomeTeam].append(HTSTF)
        stf_dict[ds.iloc[i].AwayTeam].append(ATSTF)
        sc_dict[ds.iloc[i].HomeTeam].append(HTSC)
        sc_dict[ds.iloc[i].AwayTeam].append(ATSC)
        stc_dict[ds.iloc[i].HomeTeam].append(HTSTC)
        stc_dict[ds.iloc[i].AwayTeam].append(ATSTC)
        
    # Create dataframes where rows are teams and cols are matchweek.
    GoalsScored = pd.DataFrame(data=gs_dict, index = [i for i in range(1, matchweeks+1)]).T
    GoalsConceded = pd.DataFrame(data=gc_dict, index = [i for i in range(1, matchweeks+1)]).T
    GoalDifference = pd.DataFrame(data=gd_dict, index = [i for i in range(1, matchweeks+1)]).T
    ShotsFor = pd.DataFrame(data=sf_dict, index = [i for i in range(1, matchweeks+1)]).T
    ShotsTargetFor = pd.DataFrame(data=stf_dict, index = [i for i in range(1, matchweeks+1)]).T
    ShotsConceded = pd.DataFrame(data=sc_dict, index = [i for i in range(1, matchweeks+1)]).T
    ShotsTargetConceded = pd.DataFrame(data=stc_dict, index = [i for i in range(1, matchweeks+1)]).T
    GoalsScored[0] = 0
    GoalsConceded[0] = 0
    GoalDifference[0] = 0
    ShotsFor[0] = 0
    ShotsTargetFor[0] = 0
    ShotsConceded[0] = 0
    ShotsTargetConceded[0] = 0

    # Aggregate to get uptil that point
    for i in range(2, matchweeks + 1):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
        GoalDifference[i] = GoalDifference[i] + GoalDifference[i-1]
        ShotsFor[i] = ShotsFor[i] + ShotsFor[i-1]
        ShotsTargetFor[i] = ShotsTargetFor[i] + ShotsTargetFor[i-1]
        ShotsConceded[i] = ShotsConceded[i] + ShotsConceded[i-1]
        ShotsTargetConceded[i] = ShotsTargetConceded[i] + ShotsTargetConceded[i-1]

    return GoalsScored, GoalsConceded, GoalDifference, ShotsFor, ShotsTargetFor, ShotsConceded, ShotsTargetConceded


# Get Home and Away stats for each team over the course of the season
def get_home_away(ds):
    homegroup = ds.groupby('HomeTeam')
    awaygroup = ds.groupby('AwayTeam')

    home_wins = homegroup['HomeWins'].sum()
    away_wins = awaygroup['AwayWins'].sum()
    home_draws = homegroup['HomeDraws'].sum()
    away_draws = awaygroup['AwayDraws'].sum()
    home_loss = homegroup['HomeLosses'].sum()
    away_loss = awaygroup['AwayLosses'].sum()
    home_points = homegroup['HomePoints'].sum()
    away_points = awaygroup['AwayPoints'].sum()

    home_gs = homegroup['FTHG'].sum()
    away_gs = awaygroup['FTAG'].sum()
    home_gc = homegroup['FTAG'].sum()
    away_gc = awaygroup['FTHG'].sum()

    home_sf = homegroup['HS'].sum()
    away_sf = awaygroup['AS'].sum()
    home_sc = homegroup['AS'].sum()
    away_sc = awaygroup['HS'].sum()
    home_stf = homegroup['HST'].sum()
    away_stf = awaygroup['AST'].sum()

    H_stats = pd.DataFrame(data = {'HW':home_wins, 'HD':home_draws, 'HL':home_loss, 'HP':home_points, 'HGS':home_gs, 'HGC':home_gc, 'HSF':home_sf, 'HSC':home_sc, 'HSTF':home_stf, 'HSTC':home_stf}, index=home_wins.index)
    A_stats = pd.DataFrame(data = {'AW':away_wins, 'AD':away_draws, 'AL':away_loss, 'AP':away_points, 'AGS':away_gs, 'AGC':away_gc, 'ASF':away_sf, 'ASC':away_sc, 'ASTF':away_stf, 'ASTC':away_stf}, index=away_wins.index)
    HA_stats = pd.concat([H_stats, A_stats], axis=1)
    return HA_stats



# To keep track of league table throughout season
def get_league_table(SeasonPoints, SeasonWins, SeasonDraws, SeasonLoss, SeasonGoalsFor, SeasonGoalsAgainst, SeasonGoalDifference):

    table = pd.DataFrame(columns=['Pos','Team','MP','W','D','L','GF','GA','GD','Pts'])
    table['Pts'] = SeasonPoints
    table['Team'] = SeasonPoints.index
    table['MP'] = SeasonWins + SeasonDraws + SeasonLoss
    table['W'] = SeasonWins
    table['D'] = SeasonDraws
    table['L'] = SeasonLoss
    table['GF'] = SeasonGoalsFor
    table['GA'] = SeasonGoalsAgainst
    table['GD'] = SeasonGoalDifference
    table = table.set_index('Team')
    table = table.sort_values(by=['Pts','GD','GF'], ascending=False)
    table['Pos'] = pd.Series(range(1,21), index=table.index)
    
    return table



# Put together the previous functions to calculate all the stats
def get_stats(ds):
    GS, GC, GD, SF, STF, SC, STC = cum_goalstats(ds)
    W, D, L, P, PR, F5 = cum_results(ds)

    j = 0
    MW = []
    
    HW = []
    AW = []
    HD = []
    AD = []
    HL = []
    AL = []
    HP = []
    AP = []

    HPR = []
    APR = []
    HF5 = []
    AF5 = []

    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []
    HTGD = []
    ATGD = []

    HTSF = []
    ATSF = []
    HTSTF = []
    ATSTF = []
    HTSC = []
    ATSC = []
    HTSTC = []
    ATSTC = []

    HPR = []
    APR = []
    HF5 = []
    AF5 = []

    for i in range(len(ds)):
        ht = ds.iloc[i].HomeTeam
        at = ds.iloc[i].AwayTeam

        MW.append(j+1)

        HW.append(W.loc[ht][j])
        AW.append(W.loc[at][j])
        HD.append(D.loc[ht][j])
        AD.append(D.loc[at][j])
        HL.append(L.loc[ht][j])
        AL.append(L.loc[at][j])
        HP.append(P.loc[ht][j])
        AP.append(P.loc[at][j])

        HPR.append(PR.loc[ht][j])
        APR.append(PR.loc[at][j])
        HF5.append(F5.loc[ht][j])
        AF5.append(F5.loc[at][j])

        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        HTGD.append(GD.loc[ht][j])
        ATGD.append(GD.loc[at][j])

        HTSF.append(SF.loc[ht][j])
        ATSF.append(SF.loc[at][j])
        HTSTF.append(STF.loc[ht][j])
        ATSTF.append(STF.loc[at][j])
        HTSC.append(SC.loc[ht][j])
        ATSC.append(SC.loc[at][j])
        HTSTC.append(STC.loc[ht][j])
        ATSTC.append(STC.loc[at][j])
        
        if ((i + 1)% 10) == 0:
            j = j + 1
        
    ds['MW'] = MW

    ds['HP'] = HP
    ds['AP'] = AP
    ds['Pdiff'] = ds['HP'] - ds['AP']

    ds['HW'] = HW
    ds['AW'] = AW
    ds['HD'] = HD
    ds['AD'] = AD
    ds['HL'] = HL
    ds['AL'] = AL

    ds['HTGS'] = HTGS
    ds['ATGS'] = ATGS
    ds['HTGC'] = HTGC
    ds['ATGC'] = ATGC
    ds['HTGD'] = HTGD
    ds['ATGD'] = ATGD

    ds['HTSF'] = HTSF
    ds['ATSF'] = ATSF
    ds['HTSTF'] = HTSTF
    ds['ATSTF'] = ATSTF
    ds['HTSC'] = HTSC
    ds['ATSC'] = ATSC
    ds['HTSTC'] = HTSTC
    ds['ATSTC'] = ATSTC

    ds['HPR'] = HPR
    ds['APR'] = APR
    ds['HF5'] = HF5
    ds['AF5'] = AF5

    ds.drop(['HomeWins','AwayWins','HomeDraws','AwayDraws','HomeLosses','AwayLosses','HomePoints','AwayPoints'], axis=1, inplace=True)
    
    return ds

# Single function to convert any raw data to engineered data
def engg(ds):
    req_cols = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR', 'HS', 'AS', 'HST', 'AST']
    ds = ds[req_cols]
    stats_ds = get_stats(ds)
    return stats_ds



#########################################################################################################

# Load up the data
data = []
data.append(pd.read_csv('data/epl0001.csv'))
data.append(pd.read_csv('data/epl0102.csv'))
data.append(pd.read_csv('data/epl0203.csv'))
data.append(pd.read_csv('data/epl0304.csv'))
data.append(pd.read_csv('data/epl0405.csv', encoding="windows-1252"))
data.append(pd.read_csv('data/epl0506.csv'))
data.append(pd.read_csv('data/epl0607.csv'))
data.append(pd.read_csv('data/epl0708.csv'))
data.append(pd.read_csv('data/epl0809.csv'))
data.append(pd.read_csv('data/epl0910.csv'))
data.append(pd.read_csv('data/epl1011.csv'))
data.append(pd.read_csv('data/epl1112.csv'))
data.append(pd.read_csv('data/epl1213.csv'))
data.append(pd.read_csv('data/epl1314.csv'))
data.append(pd.read_csv('data/epl1415.csv'))
data.append(pd.read_csv('data/epl1516.csv'))
data.append(pd.read_csv('data/epl1617.csv'))
data.append(pd.read_csv('data/epl1718.csv'))
data.append(pd.read_csv('data/epl1819.csv'))
data.append(pd.read_csv('data/epl1920.csv'))
data.append(pd.read_csv('data/epl2021.csv'))
data.append(pd.read_csv('data/epl2122.csv'))
data.append(pd.read_csv('data/epl2223.csv'))
data.append(pd.read_csv('data/epl2324.csv'))

# for i in range(24):
#     data[i].to_csv('engg_data/epl' + str("{:02d}".format(i)) + str("{:02d}".format(i+1)) + '.csv', index=False)


#########################################################################################################

# Functions to display the plots and results

team_abrev = {'Arsenal': 'Ars',
            'Aston Villa': 'Avl',
            'Birmingham': 'Bir',
            'Blackburn': 'Blb',
            'Blackpool': 'Blp',
            'Bolton': 'Bol',
            'Bournemouth': 'Bou',
            'Bradford': 'Bra',
            'Brentford': 'Bre',
            'Brighton': 'Bha',
            'Burnley': 'Bur',
            'Cardiff': 'Car',
            'Charlton': 'Cha',
            'Chelsea': 'Che',
            'Coventry': 'Cov',
            'Crystal Palace': 'Cry',
            'Derby': 'Der',
            'Everton': 'Eve',
            'Fulham': 'Ful',
            'Huddersfield': 'Hud',
            'Hull': 'Hul',
            'Ipswich': 'Ips',
            'Leeds': 'Lee',
            'Leicester': 'Lei',
            'Liverpool': 'Liv',
            'Luton': 'Lut',
            'Man City': 'Mci',
            'Man United': 'Mun',
            'Middlesbrough': 'Mid',
            'Newcastle': 'New',
            'Norwich': 'Nor',
            'Nott\'m Forest': 'Nfo',
            'Portsmouth': 'Por',
            'QPR': 'Qpr',
            'Reading': 'Rea',
            'Sheffield United': 'Shu',
            'Southampton': 'Sou',
            'Stoke': 'Stk',
            'Sunderland': 'Sun',
            'Swansea': 'Swa',
            'Tottenham': 'Tot',
            'Watford': 'Wat',
            'West Brom': 'Wba',
            'West Ham': 'Whu',
            'Wigan': 'Wig',
            'Wolves': 'Wol'
              }


# Plots the league table
def plot_table(table):
    st.dataframe(table)


# Plots a line chart of points progression for all teams together over the season
def plot_season(points):
    srt_p = points.T.iloc[:totalmw, :].sort_values(by=totalmw, axis=1, ascending=False)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(srt_p)
    plt.title('Season Progression')
    plt.xlabel('Matchweek')
    plt.ylabel('Points')
    plt.legend(srt_p.columns, bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)


# Plots the distribution of a stat among teams over the course of the season
def plot_pie(stat, statname):
    sorted_stat = stat.sort_values(ascending=False)
    team_labels = [team_abrev[i] for i in sorted_stat.index]

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.pie(sorted_stat, labels=team_labels, autopct='%1.1f%%', startangle=90, explode=(0.2,0.15,0.1,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), shadow=True)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # title = 'PieChart of ' + statname
    # plt.title(title)
    plt.tight_layout()

    st.pyplot(fig1)


# Plots the performance of teams at Home and Away over the course of the season
def plot_bar(stat, hstat, statname):
    sorted_stat = stat.sort_values(ascending=False)
    team_labels = [team_abrev[i] for i in sorted_stat.index]
    sorted_hstat = hstat.sort_values(ascending=False)

    hlabel = 'Home ' + statname
    alabel = 'Away ' + statname
    title = 'H/A BarChart of ' + statname

    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.bar(team_labels, sorted_stat, label=alabel)
    ax.bar(team_labels, sorted_hstat, label=hlabel)
    ax.set_xlabel('Teams')
    ax.set_ylabel(statname)
    ax.set_title(title)
    ax.legend()

    # Display the ratio of home/away goals on each bar
    i=0
    for team, v in sorted_stat.items():
        ax.text(i-0.35, v+1, str(round(hstat[team]/v, 2)), color='black')
        i = i + 1

    st.pyplot(fig2)


# Plots the distribution of a stat across matchweeks
def plot_mw_stats(data, statname, mode, annot):

    mwstat_dict = {'HomeGoals': 'FTHG', 'HomeShots': 'HS', 'HomeShots on Target': 'HST',
                   'AwayGoals': 'FTAG', 'AwayShots': 'AS', 'AwayShots on Target': 'AST'}
    
    if(statname == 'Goals' or statname == 'Shots' or statname == 'Shots on Target'):
        if mode != 'Overall':
            key = mwstat_dict[mode + statname]
            mw_stat = data.groupby('MW')[key].sum()
        if mode == 'Overall':
            key1 = mwstat_dict['Home' + statname]
            key2 = mwstat_dict['Away' + statname]
            mw_stat = data.groupby('MW')[key1].sum() + data.groupby('MW')[key2].sum()
    else:
        if mode != 'Overall':
            key = mode + statname
            mw_stat = data.groupby('MW')[key].sum()
        if mode == 'Overall':
            key1 = 'Home' + statname
            key2 = 'Away' + statname
            mw_stat = data.groupby('MW')[key1].sum() + data.groupby('MW')[key2].sum()

    fig3, ax = plt.subplots(figsize=(12, 6))
    ax.bar(mw_stat.index, mw_stat)
    ax.set_xlabel('Matchweek')
    ax.set_ylabel(statname)
    title = 'Matchweek-wise Distribution of ' + statname
    ax.set_title(title)

    # If annot toggled, display the value of each bar
    if annot:
        i=0
        for v in mw_stat.values:
            ax.text(i+0.6, v+1, str(v), color='black')
            i = i + 1
        ax.set_xticks(list(range(1, totalmw+1)))

    st.pyplot(fig3)


# Plots the correlation between 2 stats
def plot_scatter(data, stat1, stat2, statname1, statname2):
    fig4 = plt.figure(figsize=(12, 6))
    plt.scatter(data[stat2], data[stat1], alpha=0.5)
    plt.plot(np.unique(data[stat2]), np.poly1d(np.polyfit(data[stat2], data[stat1], 1))(np.unique(data[stat2])))
    # plt.plot(data[stat2], np.poly1d(np.polyfit(data[stat2], data[stat1], 1))(data[stat2]))
    title = statname1 + ' vs ' + statname2
    plt.title(title)
    plt.xlabel(statname2)
    plt.ylabel(statname1)

    r, p = stats.pearsonr(data[stat2], data[stat1])
    plt.annotate('r = {:.2f}'.format(r), xy=(0.47, 0.9), xycoords='axes fraction')
    
    plt.tight_layout()
    st.pyplot(fig4)
    
    

#########################################################################################################

# Web app code
    
st.set_page_config(page_icon="img/pl.jpg", page_title="EPL Viz", layout="wide")

st.write("""
         # ⚽ EPL Viz ✨
         A visual deep dive into the last *24 years* of the `English Premier League`! 🕵️‍♂️
         """)
st.write('---')

st.sidebar.header('Links')
st.sidebar.link_button('GitHub Repo', 'https://github.com/saranggalada/EDA-English-Premier-League-24yr')
st.sidebar.link_button('Data Source', 'https://www.football-data.co.uk/')
# st.sidebar.link_button('Author', 'https://www.linkedin.com/in/saranggalada')
st.sidebar.markdown("---\n*Copyright (c) 2024: Sarang Galada*")

season = st.selectbox('Select EPL Season', ('2023-24 season','2022-23 season','2021-22 season',
                                     '2020-21 season','2019-20 season','2018-19 season',
                                     '2017-18 season','2016-17 season','2015-16 season',
                                     '2014-15 season','2013-14 season','2012-13 season',
                                     '2011-12 season','2010-11 season','2009-10 season',
                                     '2008-09 season','2007-08 season','2006-07 season',
                                     '2005-06 season','2004-05 season','2003-04 season',
                                     '2002-03 season','2001-02 season','2000-01 season'))

seasons = ['2000-01', '2001-02', '2002-03', '2003-04', '2004-05', '2005-06', 
           '2006-07', '2007-08', '2008-09', '2009-10', '2010-11', '2011-12',
           '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', 
           '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']


# Run these 2 lines only if engg_data csv files AREN'T already created
ds = data[seasons.index(season[:7])]
engg_ds = engg(ds)

totalmw = int(len(ds)/10)
w, d, l, p, pr, f5 = cum_results(ds)
gs, gc, gd, sf, stf, sc, stc = cum_goalstats(ds)
ha_stats = get_home_away(ds)
table = get_league_table(p[totalmw], w[totalmw], d[totalmw], l[totalmw], gs[totalmw], gc[totalmw], gd[totalmw])



# Create the tabs
tabs = ['League Table', 'Season Progress Chart', 'Matchweek Analysis', 'Home/Away Performance', 'Pie Chart', 'Correlation Plots', ]
tab_list = st.tabs(tabs)

# Load up each tab with a plot
for plot_num, tab in enumerate(tab_list):
    with tab:
        # Can't get the name of the tab! Can't index key list. So convert to list and index
        plot_name = tabs[plot_num]

        stat_dict = {'Points': p, 'Wins': w, 'Draws': d, 'Losses': l, 
                     'Goals Scored': gs, 'Goals Conceded': gc, 
                     'Shots For': sf, 'Shots on Target For': stf, 
                     'Shots Conceded': sc, 'Shots on Target Conceded': stc}
        
        hstat_dict = {'Points': ha_stats['HP'], 'Wins': ha_stats['HW'], 'Draws': ha_stats['HD'], 'Losses': ha_stats['HL'], 
                      'Goals Scored': ha_stats['HGS'], 'Goals Conceded': ha_stats['HGC'], 
                      'Shots For': ha_stats['HSF'], 'Shots on Target For': ha_stats['HSTF'], 
                      'Shots Conceded': ha_stats['HSC'], 'Shots on Target Conceded': ha_stats['HSTC']}
        
        agg_stats = {'Home Points': 'HP', 'Home Wins': 'HW', 'Home Draws': 'HD', 'Home Losses': 'HL',
                     'Home Goals Scored': 'HTGS', 'Home Goals Conceded': 'HTGC', 
                     'Home Shots For': 'HTSF', 'Home Shots on Target For': 'HTSTF',
                     'Home Shots Conceded': 'HTSC', 'Home Shots on Target Conceded': 'HTSTC',
                     'Away Points': 'AP', 'Away Wins': 'AW', 'Away Draws': 'AD', 'Away Losses': 'AL',
                     'Away Goals Scored': 'ATGS', 'Away Goals Conceded': 'ATGC', 
                     'Away Shots For': 'ATSF', 'Away Shots on Target For': 'ATSTF',
                     'Away Shots Conceded': 'ATSC', 'Away Shots on Target Conceded': 'ATSTC'}

        st.subheader(plot_name)
        if plot_name == 'League Table':
            plot_table(table)

        elif plot_name == 'Season Progress Chart':
            plot_season(p)

        elif plot_name == 'Matchweek Analysis':
            statname = st.selectbox('Statistic', ('Goals', 'Points', 'Wins', 'Draws', 'Losses', 'Shots', 'Shots on Target'))
            mode = st.radio('Mode', ('Overall', 'Home', 'Away'), horizontal=True)
            annot = st.toggle('Annotations', True)
            plot_mw_stats(engg_ds, statname, mode, annot)

        elif plot_name == 'Home/Away Performance':
            statname = st.selectbox('H/A Statistic', ('Points', 'Wins', 'Draws', 'Losses', 'Goals Scored', 'Goals Conceded', 'Shots For', 'Shots on Target For', 'Shots Conceded', 'Shots on Target Conceded'))
            stat = stat_dict[statname]
            plot_bar(stat[totalmw], hstat_dict[statname], statname)

        elif plot_name == 'Pie Chart':
            statname = st.selectbox('Stat', ('Shots For', 'Shots on Target For', 'Goals Scored', 'Points', 'Goals Conceded', 'Shots Conceded', 'Shots on Target Conceded', 'Wins', 'Draws', 'Losses',))
            stat = stat_dict[statname]
            plot_pie(stat[totalmw], statname)

        elif plot_name == 'Correlation Plots':
            statname1 = st.selectbox('Stat 1', ('Home Points', 'Home Wins', 'Home Draws', 'Home Losses', 'Home Goals Scored', 'Away Points', 'Away Wins', 'Away Draws', 'Away Losses', 'Away Goals Scored', 'Home Shots For', 'Home Shots on Target For', 'Away Shots For', 'Away Shots on Target For', 'Home Goals Conceded', 'Home Shots Conceded', 'Home Shots on Target Conceded', 'Away Goals Conceded', 'Away Shots Conceded', 'Away Shots on Target Conceded'))
            statname2 = st.selectbox('Stat 2', ('Home Goals Scored', 'Home Shots For', 'Home Shots on Target For', 'Away Goals Scored', 'Away Shots For', 'Away Shots on Target For', 'Home Goals Conceded', 'Home Shots Conceded', 'Home Shots on Target Conceded', 'Away Goals Conceded', 'Away Shots Conceded', 'Away Shots on Target Conceded', 'Home Points', 'Home Wins', 'Home Draws', 'Home Losses', 'Away Points', 'Away Wins', 'Away Draws', 'Away Losses'))
            plot_scatter(engg_ds, agg_stats[statname1], agg_stats[statname2], statname1, statname2)
