import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import scipy.integrate
import scipy.ndimage.interpolation  # shift function
import base64


st.title('Refugee Camp Epidemic Model')


st.sidebar.title('Parameters')
days = st.sidebar.slider('days of simulation', 0, 250, 75)
population = st.sidebar.slider('Population size', 0, 100_000, 20_000)
intervention_day = st.sidebar.slider('Intervention day', 0, days, 10)
inititalr0 = st.sidebar.slider('InitialR0', 0.0, 20.0, 14.0, 0.1)
new_r0 = st.sidebar.slider('New R0', 0.0, 20.0, 7.0, 0.1)
#remove_people = st.sidebar.slider('Number of people removed from camp', 0, population)
fatality_rate = st.sidebar.slider('Fatality_rate in %', 1.0, 100.0, 3.0, 0.1)


"""


"""

E0 = 1  # exposed at initial time step
days0 = intervention_day
daysTotal = days  # total days to model


r0 = inititalr0  # https://en.wikipedia.org/wiki/Basic_reproduction_number
r1 = new_r0  # reproduction number after quarantine measures - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3539694
# it seems likely that measures will become more restrictive if r1 is not small enough

timePresymptomatic = 2.5
sigma = 1.0 / (5.2 - timePresymptomatic)  # The rate at which an exposed person becomes infectious.
# for SEIR: generationTime = 1/sigma + 0.5 * 1/gamma = timeFromInfectionToInfectiousness + timeInfectious  https://en.wikipedia.org/wiki/Serial_interval
generationTime = 4.6
# The rate an infectious is not recovers and moves into the resistant phase. Note that for the model it only means he does not infect anybody any more.
gamma = 1.0 / (2.0 * (generationTime - 1.0 / sigma))

# The parameter controlling how often a susceptible-infected contact results in a new infection.
beta0 = r0 * gamma
beta1 = r1 * gamma  # beta0 is used during days0 phase, beta1 after days0

Infectious_period = 10
Incubation_period = 5
sigma2 = 1/Incubation_period
gamma2 = 1/Infectious_period

Interevention = 0

age0_1 = 60/100 * population
age18_39 = 35/100 * population
age50 = 5 / 100 * population


def model(Y, x, N, beta0, days0, beta1, gamma, sigma):

    S, E, I, R = Y

    beta = beta0 if x < days0 else beta1

    dS = - beta * S * I / N
    dE = beta * S * I / N - sigma2 * E
    dI = sigma2 * E - gamma2 * I
    dR = gamma2 * I
    return dS, dE, dI, dR


def solve(model, population, E0, beta0, days0, beta1, gamma, sigma):

    X = np.arange(daysTotal)  # time steps array
    N0 = population - E0, E0, 0, 0  # S, E, I, R at initial step

    y_data_var = scipy.integrate.odeint(model, N0, X, args=(
        population, beta0, days0, beta1, gamma, sigma))

    S, E, I, R = y_data_var.T  # transpose and unpack
    return X, S, E, I, R  # note these are all arrays


X, S, E, I, R = solve(model, population, E0, beta0, days0, beta1, gamma, sigma)


# estimate deaths from recovered
infectionFatalityRateA = fatality_rate/100
D = np.arange(daysTotal)
RPrev = 0
DPrev = 0
for i, x in enumerate(X):
    IFR = infectionFatalityRateA  # if U[i] <= intensiveUnits else infectionFatalityRateB
    D[i] = DPrev + IFR * (R[i] - RPrev)
    RPrev = R[i]
    DPrev = D[i]


data = pd.DataFrame({'Days': X, 'Susceptible': S, 'Exposed': E,
                     'Infectious': I, 'Removed': R, 'Fatalities': D})
cols = ["Susceptible", "Exposed", "Infectious", "Removed"]


fatality = pd.DataFrame({'Days': X, 'Fatalities': D})

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


new_df = data.melt('Days', var_name='Status', value_name='Population')
new_df = new_df[new_df['Status'] != 'Susceptible']

#st_ms = st.multiselect("Select different columns", data.columns.tolist(), default=cols)


chart2 = alt.Chart(new_df).mark_bar(size=6.5).encode(
    y='Population',
    x='Days',
    color=alt.Color('Status', scale=alt.Scale(scheme='tableau10')),
    tooltip='Population'
).properties(width=660, height=340
             ).configure_axis(grid=False
                              ).configure_view(strokeWidth=0

                                               )

st.altair_chart(chart2)

# st.bar_chart(data[st_ms])


chart = alt.Chart(fatality).mark_bar().encode(
    y='Fatalities', x='Days'
).properties(width=580, height=340
             ).configure_axis(grid=False
                              ).configure_view(strokeWidth=0

                                               )

st.altair_chart(chart)


csv = data.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
st.markdown(href, unsafe_allow_html=True)
