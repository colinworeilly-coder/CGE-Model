# regcge model specification

This document lists every equation of the model, explains the departures from
Texas-STAMP (Beacon Hill Institute, 2004), and gives the correspondence with
STAMP's equation numbers.

## 1. Overview

`regcge` is a **recursive-dynamic computable general equilibrium model of a
sub-national (state) economy**. It is calibrated to a Social Accounting Matrix
(SAM) and designed to analyse tax policy. Each year is a static equilibrium of:

* competitive producers with nested CES technologies;
* utility-maximising households of several income classes, who choose
  consumption, saving and labor supply;
* governments (any number, e.g. federal, state and local) with a generic set of
  tax instruments and switchable budget closures;
* the rest of the world (other states plus other countries), linked through
  Armington imports, regional export demand, factor flows and transfers.

Between years, capital accumulates through investment that responds to
after-tax returns, and the population grows and migrates.

| | Texas-STAMP | regcge |
|---|---|---|
| Platform | GAMS, Excel front end | Python. CasADi for automatic differentiation; Newton/IPOPT solvers |
| Household demand | Log-linear demand functions in prices and income (eq. 3) | ELES over goods and saving, CES with leisure. Utility-consistent |
| Labor supply | Log-linear participation function of wage, taxes and transfers (eq. 6) | Derived from utility. Elasticity calibrated to a target |
| Welfare | Disposable income, per-capita income | **Exact equivalent variation** by class; MEB/MCF |
| Income taxes | Average rates with deductibility (eq. 21) | Linear schedule with **marginal and average** rates, SALT deductibility, **federal offset** |
| Tax coverage | 23 hard-coded tax sectors | Any number of instruments declared in data; 5 base types |
| Government closure | Spending follows revenue | Per government: spending, deficit, a tax rate (equal yield) or transfers adjust |
| Exports | Constant-elasticity export demand | Same, optionally with CET transformation, or price taking |
| Investment | Relative after-tax return (eq. 19) | Tobin's-q form that converges to a steady state |
| Migration | Instantaneous (eq. 7) | Partial adjustment; real income or utility as the driver |
| Baseline | Steady-state path from growth rates | Exact balanced-growth path, by construction |
| Diagnostics | – | Closure squareness and structural-rank checks; Walras check; tests |

## 2. Notation and conventions

Sets:

| Set | Meaning |
|---|---|
| $a, c \in A$ | sectors = commodities (one-to-one) |
| $f \in F$ | factors: labor types $l \in L$, one capital factor $K$, and fixed factors $x \in X$ |
| $h \in H$ | household income classes |
| $g \in G$ | governments |
| $k \in T$ | tax instruments, split by base type: $T^S$ (sales), $T^O$ (output), $T^F$ (factor), $T^P$ (property), $T^I$ (income) |
| $t$ | years $0,\dots,T$ |

**Calibrated share form.** Every CES, CET and Cobb-Douglas function is written
relative to its benchmark point (Rutherford 1998):

$$c(p) = \Big[\sum_i \theta_i (p_i/p_i^0)^{1-\sigma}\Big]^{1/(1-\sigma)},\qquad
x_i = x_i^0\,\frac{Q}{Q^0}\Big(\frac{c(p)}{p_i/p_i^0}\Big)^{\sigma},$$

where $\theta_i$ are benchmark value shares. Leontief ($\sigma=0$) and
Cobb-Douglas ($\sigma=1$) are the limiting cases. CET revenue functions use
the exponent $1+\eta$. Zero benchmark flows drop out automatically.

**Units.** All benchmark prices are one, except factor user costs, which
include factor taxes. Labor is measured in efficiency units, so the benchmark
wage is one. Capital is measured in units of the capital good, with benchmark
price $PK^0=1$. Money values are in SAM units, for example \$ billions.

**Growth indices.** $GI_t=(1+g)^t$ for aggregates, $PCI_t=(1+a)^t$ for
per-capita quantities, and $(1+n)^t$ for population, with $1+g=(1+n)(1+a)$.

## 3. Equations (one period)

Purchaser prices include sales-tax wedges. For example, households pay
$PH_{c,h}=PQ_c(1+\sum_{k\in T^S} TS_k\,ts^h_{k,c,h})$. Here $TS_k$ is the
instrument's scale variable (normally fixed at one) and $ts^h$ are
time-varying rate parameters. Intermediate, investment and government
purchases are treated analogously.

### 3.1 Production (STAMP eqs. 9–12)

* **Factor user costs:**
  * labor: $UC_{l,a}=W_l(1+tf_{l,a})$;
  * capital: $UC_{K,a}=RK_a(1+tf_{K,a})+tprop_a\,PK_a$;
  * fixed factors: $UC_{x,a}=RFX_{x,a}(1+tf_{x,a})$.
* **Price of value added:** $PVA_a = \frac{1}{TFP_a}\,c^{VA}_a(UC_{\cdot,a};\sigma^{VA}_a)$.
* **Factor demand:** $QF_{f,a}=QF^0_{f,a}\frac{QVA_a}{QVA^0_a}\frac{1}{TFP_a}\big(\frac{PVA_a TFP_a}{UC_{f,a}/UC^0_{f,a}}\big)^{\sigma^{VA}_a}$.
* **Intermediate-bundle price:** $PINTA_a=\sum_c io_{c,a}PQ_c(1+ts^i_{c,a})$, Leontief across commodities.
* **Zero profit (top nest):** $PX_a(1-tout_a)=c^0_a\,c^{TOP}_a(PVA_a,PINTA_a;\sigma^{TOP}_a)$, where $c^0_a = 1-tout^0_a$.
* **Top-nest demands:** $QVA_a$ and $QINTA_a$ follow from Shephard's lemma. The default $\sigma^{TOP}=0$ is Leontief, as in STAMP.

### 3.2 Trade (STAMP eqs. 14–17)

* **Armington composite:**
  * price: $PQ_c=c^{ARM}_c(PD_c,PM_c;\sigma^{A}_c)$;
  * in-state demand: $QD_c=QD^0_c\frac{QQ_c}{QQ^0_c}(PQ_c/PD_c)^{\sigma^A_c}$;
  * import demand: $QM_c=QM^0_c\frac{QQ_c}{QQ^0_c}(PQ_c/PM_c)^{\sigma^A_c}$.
* **Import prices:** $PM_c=PWM_c\cdot ER$.
* **Output transformation**, set per sector:
  * perfect transformation (STAMP): $PD=PE=PX$ and $QX=QD+QE$;
  * CET with elasticity $\eta^T$: $PX = r(PD,PE)$, and supplies follow Hotelling's lemma.
* **Export demand:** $QE_c=QE^0_c\,GI\,EXS_c\,(PE_c/(PWE_c ER))^{-\eta^E_c}$. With $\eta^E=\infty$, the state is a price taker: $PE_c=PWE_c ER$.

### 3.3 Factor markets and incomes (STAMP eqs. 13, 29, 30, 39)

* **Labor:** $(1-nr_l)\big(\sum_a QF_{l,a}+\sum_g QFG_{l,g}\big)=\sum_h \lambda_{l,h} LS_h$, where $nr_l$ is the commuter share of jobs.
* **Capital, sector-specific:** $QF_{K,a}=K_a$.
* **Capital, mobile mode:** $RK_a = RKAVG\cdot RK^0_a/\overline{RK^0}$, with $\sum_a K_a = KTOT$.
* **Fixed factors:** $QF_{x,a}=QF^0_{x,a}GI$.
* **Capital income:** $YFK=\sum_a RK_aK_a + (\text{gov. capital rentals} + \text{income from out of state})\,GI\,ER$.

### 3.4 Households (replaces STAMP eqs. 1–6)

**Income.**

$$YH_h = \sum_l\big[W_l\lambda_{l,h}LS_h + \rho_{l,h}YLROW^0_l GI\,ER\big] + \theta^K_h YFK + \sum_x \theta^X_{x,h}YFX_x + div_h YENT + \sum_g TRG_{h,g} + TRROW^0_h GI\,ER$$

with transfers indexed to population and prices,
$TRG_{h,g}=TRSCALE_g\,tr_{h,g}\,N_h\,PCI\,CPI_h$.

**Direct taxes.** For each income-tax instrument $k\in T^I$:

$$TAXH_{k,h}=TS_k\big(mtr_{k,h}\,BASE_{k,h}-A_{k,h}\,PCI\,\tfrac{N_h}{N_h^0}\,CPI_h\big),$$

$$BASE_{k,h}=\sum_{j\in\{lab,cap,tr\}} w_{k,j}\,INC_{j,h}-d_h\sum_{k'\in D_k}PAID_{k',h},$$

* $D_k$ is the set of taxes deductible from $k$'s base, e.g. state income and residential property taxes against the federal income tax.
* $d_h$ is the share deducted at the margin (itemizers).
* The intercept $A$ is calibrated so that the average rate matches the SAM at benchmark, and it is indexed. The schedule is therefore progressive: $mtr > avg$ whenever $A > 0$.
* The effective marginal tax rate on labor, net of deductibility interactions, is
  $MTRL_h=\mathbf 1'(I+\mathrm{diag}(m)D)^{-1}\mathrm{diag}(m)w_L$.
* Disposable income: $YD_h=YH_h-\sum_k TAXH_{k,h}$.

**Preferences.** Utility is CES between leisure and a supernumerary bundle
$Z$, with elasticity $\sigma^{LS}_h$. $Z$ is a Cobb-Douglas (ELES) aggregate of
the supernumerary quantities of every commodity, of direct out-of-state
purchases and of saving, with marginal budget shares $\beta_{i,h}$ and
subsistence quantities $\gamma_{i,h}$.

* Committed expenditure and supernumerary income: $C_h=\sum_i P_{i,h}\gamma_{i,h}N_hPCI$, and $SUP_h=YD_h-C_h$.
* Price of the bundle: $PZ_h=\prod_i (P_{i,h}/P^0_{i,h})^{\beta_{i,h}}$.
* Price of leisure (the marginal net wage): $PL_h=\bar W_h(1-MTRL_h)$.
* Full supernumerary income and the household price index:
  $FS_h=SUP_h+PL_h\,LEIS_h$, and $\Phi_h=c(PZ_h,PL_h/PL^0_h;\sigma^{LS}_h)$.

Equations:

$$U_h\,\Phi_h = FS_h,\qquad LEIS_h = U_h\tfrac{LEIS^0_h}{U^0_h}\Big(\tfrac{\Phi_h}{PL_h/PL^0_h}\Big)^{\sigma^{LS}_h},\qquad Z_h=U_h\tfrac{Z^0_h}{U^0_h}\Big(\tfrac{\Phi_h}{PZ_h}\Big)^{\sigma^{LS}_h},$$

$$LS_h = N_h\,PCI\,e_h - LEIS_h,\qquad QH_{c,h}=\gamma_{c,h}N_hPCI+\beta_{c,h}\frac{PZ_hZ_h}{PH_{c,h}},\qquad SAVH_h=PSAV\gamma_{s,h}N_hPCI+\beta_{s,h}PZ_hZ_h.$$

The household budget holds identically, because $PZ\cdot Z+PL\cdot LEIS=\Phi U$.

The CPI is a Laspeyres index: $CPI_h=\sum_c w^0_{c,h}PH_{c,h}/PH^0_{c,h}+w^0_{r,h}PROW$.

**Calibration.**
* $\beta$ and $\gamma$ come from benchmark budget shares, income elasticities (default 1) and a Frisch parameter (default $-1$, which is Cobb-Douglas, as in STAMP).
* The time endowment is $e_h N_h = \psi_h LS^0_h$, with default $\psi=1.75$.
* $\sigma^{LS}_h$ is solved in closed form to reproduce a target uncompensated labor-supply elasticity $\varepsilon_h$ with respect to the marginal net wage:

$$\sigma^{LS}=1-\frac{1-PL\,E/FS-\varepsilon\,LS/LEIS}{1-s_L},\qquad s_L=PL\cdot LEIS/FS.$$

  Setting labor supply to exogenous sets $\psi=1$.

### 3.5 Migration (STAMP eqs. 7–8)

$$N_h=\bar N_h\Big(\frac{N_{h,t-1}}{\bar N_{h,t-1}}\Big)^{1-\lambda_h}\Big(\frac{X_h}{\bar X_h}\Big)^{\lambda_h\eta^m_h}$$

Here $X_h=YD_h/(N_hCPI_h)$ by default (real disposable income per household)
or $U_h/N_h$, and bars denote values on the baseline path. On the baseline,
$N$ is fixed and $\bar X$ is free; in scenarios, $\bar X$ is fixed and $N$ is
free. With $\lambda=1$ migration is instantaneous, as in STAMP.

### 3.6 Taxes (STAMP eqs. 21–22)

Revenue by instrument, $TAXR_k=TS_k\times$ the base times the rate, with:

* **sales:** household + intermediate + investment + government purchases, each at its own rate by commodity and purchaser;
* **output:** $\sum_a tout_{k,a}PX_aQX_a$;
* **factor:** $\sum_a tf_{k,a}\times$ factor payments + $\sum_g tfg_{k,g}W\,QFG$;
* **property:** $\sum_a tprop_{k,a}PK_aK_a$;
* **income:** $\sum_h TAXH_{k,h}$.

### 3.7 Governments and enterprises (STAMP eqs. 22–27)

* **Receipts:** $YG_g=\sum_k sh_{k,g}TAXR_k+\sum_{g'}TRGG_{g,g'}+\theta^K_gYFK+div_gYENT+ROWG^0_gGI\,ER$.
* **Spending:** a fixed-proportion bundle of goods and labor at real level $QGL_g$, plus transfers to households, to other governments and to enterprises, plus saving:
  $SG_g = YG_g - [\sum_c PG_{c,g}qg_{c,g}QGL_g+\sum_l W_l(1+tfg_{l,g})qf_{l,g}QGL_g+\ldots]$.
* **Intergovernmental transfers:** a share of the payer's revenue ($\mu$, as in STAMP) or fixed in real terms (default for federal grants).
* **Enterprise:**
  * income: $YENT=\theta^K_{ENT}YFK+\ldots$;
  * dividends to households, governments and out-of-state owners are fixed shares;
  * retained earnings are saved.

### 3.8 Investment and capital (STAMP eqs. 18–20)

* **Investment by destination:** $IND_a = INVSCALE\,(g+\delta_a)\,K_a\,\Big[\frac{RK_a/PK_a}{RK^0_a/PK^0_a}\Big]^{\eta^I_a}$.
* **Demand for investment goods:** $QINV_c=\sum_a B_{c,a}IND_a$, and the price of capital goods is $PK_a=\sum_c B_{c,a}PQ_c(1+ts^v_c)$.
* **Accumulation:** $K_{a,t+1}=(1-\delta_a)K_{a,t}+IND_{a,t}$.
* **Base-year capital stock:** $K^0_a=IND^0_a/(g+\delta_a)$, so the baseline is a balanced-growth path.

### 3.9 Market clearing and closure (STAMP eqs. 29–32)

* **Commodities:** $QQ_c=\sum_a QINT_{c,a}+\sum_h QH_{c,h}+\sum_g QG_{c,g}+QINV_c$.
* **Rest of world:** $SROW$ equals payments to the rest of the world (imports, direct purchases, non-resident wages, out-of-state capital income, dividends and transfers) minus receipts (exports, residents' out-of-state factor income, transfers).
* **Saving–investment:** $\sum_a PK_aIND_a=\sum_hSAVH_h+SENT+\sum_gSG_g+SROW+WALRAS$. By Walras' law $WALRAS=0$ at every solution, and this is checked.

## 4. Closures

All closures fix or free variables (see `docs/closures_scenarios.md`). The default is:
* the rest-of-world price level (ER) is the numeraire;
* open capital market: investment follows returns and ROW saving adjusts;
* FED: deficit adjusts; STATE and LOCAL: spending adjusts (STAMP);
* sector-specific capital within a period;
* endogenous labor supply and migration.

## 5. Dynamics

The baseline is the balanced-growth path implied by the SAM:
* every real quantity grows at $g$;
* prices are constant;
* households grow at $n$.

A scenario re-solves each year with the policy applied. Variables that are
exogenous under the scenario closure take their baseline values unless
shocked, so a scenario without shocks reproduces the baseline exactly.

## 6. Welfare

Per household class, the expenditure function is
$e(p,u)=C(p)+\Phi(p)\,u$ per household. The equivalent variation in year $t$,
at baseline prices of that year, is

$$EV_{h,t}=\Phi_h(p^{base}_t)\Big(\frac{U^{s}_{h,t}}{N^{s}_{h,t}}-\frac{U^{b}_{h,t}}{N^{b}_{h,t}}\Big)N^{b}_{h,t}.$$

It refers to the original residents; migrants are excluded. The welfare
measure covers within-year consumption, saving and leisure (myopic households).
Present values use `discount_rate`.

**Marginal excess burden.** The rates of one instrument are raised by 1% and
the revenue is returned lump-sum through the collecting government's
transfers. Then $MEB=-\sum_hEV_h/\Delta R$ and $MCF=1+MEB$ (Ballard, Shoven &
Whalley 1985). In a regional model with finite export-demand elasticities and
out-of-state factor ownership, the MCF of some taxes is below one in the short
run, because part of the burden is exported. With near price-taking trade and
after capital adjusts, MCFs exceed one.

## 7. Correspondence with STAMP equations

| STAMP eq. | Content | regcge |
|---|---|---|
| 1 | household gross factor income | `EQ_YH` (labor income tied to own labor supply) |
| 2 | disposable income | `EQ_YD` |
| 3 | private consumption | `EQ_QH`: ELES with leisure |
| 4 | household saving | `EQ_SAVH`: ELES saving |
| — | direct household imports | `EQ_QROWC` (pseudo-good in ELES) |
| 5 | CPI by household | `EQ_CPI` |
| 6 | labor supply (participation) | `EQ_LEIS`, `EQ_LS`: leisure demand |
| 7–8 | population, migration | `EQ_MIG` (partial adjustment), non-working households replaced by leisure |
| 9 | intermediate demand | `EQ_QINTA`, Leontief coefficients |
| 10 | CES production function | `EQ_PVA` (dual), `EQ_QF` |
| 11 | price of value added | `EQ_ZP`, `EQ_PVA` |
| 12 | factor demand | `EQ_QF` |
| 13 | factor income | `EQ_YFK`, `EQ_YFX` |
| 14 | export demand | `EQ_EXP` |
| 15 | domestic share | `EQ_ARM_D` (Armington CES) |
| 16 | import demand | `EQ_ARM_M` |
| 17 | average price | `EQ_ARM_P` |
| 18 | capital stock | `dynamics.py` accumulation |
| 19 | investment by destination | `EQ_IND` |
| 20 | investment by source | capital-coefficient matrix `B` |
| 21 | household taxes | `EQ_TAXH` |
| 22 | government income | `EQ_TAXR`, `EQ_YG` |
| 23–24 | endogenous government purchases | `QGL` under `spending_adjusts` |
| 25 | government saving | `EQ_SG` |
| 26–27 | distribution of revenue, transfers | revenue shares `shTG`, intergovernmental shares `mu` |
| 28 | state personal income | report `SPI` |
| 29 | labor market | `EQ_LAB` |
| 30 | capital market | `EQ_KAP` |
| 31–32 | goods market | `EQ_MKT` |
| 33–34, 42 | housekeeping zeros | handled by calibrated share form |
| 35–36 | federal transfers indexed | `TRG` indexed to $N$, $CPI$ |
| 37–38 | exogenous government units | `deficit_adjusts` closure |
| 39 | fixed wage differentials | efficiency units |
| 40–41 | government rental rates, capital scalar | exogenous rentals; `RKAVG` in mobile mode |
| 43 | objective function | not needed; the system is square and solved directly |

## References

* Armington, P. (1969). A theory of demand for products distinguished by place of production. *IMF Staff Papers* 16.
* Ballard, C. (2000). How many hours are in a simulated day? The effects of time endowment on the results of tax-policy simulation models. Michigan State University.
* Ballard, C., Shoven, J. & Whalley, J. (1985). General equilibrium computations of the marginal welfare costs of taxes in the United States. *AER* 75(1).
* Berck, P., Golan, E. & Smith, B. (1996). *Dynamic Revenue Analysis for California*. UC Berkeley / California Department of Finance.
* Decaluwé, B., Lemelin, A., Robichaud, V. & Maisonnave, H. (2013). *PEP-1-t: the PEP standard single-country recursive dynamic CGE model*.
* Dervis, K., de Melo, J. & Robinson, S. (1982). *General Equilibrium Models for Development Policy*. Cambridge UP.
* Golan, A., Judge, G. & Robinson, S. (1994). Recovering information from incomplete or partial multisectoral economic data. *REStat* 76(3).
* Harris, J. & Todaro, M. (1970). Migration, unemployment and development. *AER* 60(1).
* Jung, H.-S. & Thorbecke, E. (2003). The impact of public education expenditure on human capital, growth, and poverty in Tanzania and Zambia. *J. Policy Modeling* 25.
* Lluch, C. (1973). The extended linear expenditure system. *European Economic Review* 4.
* Lofgren, H., Harris, R. & Robinson, S. (2002). *A Standard CGE Model in GAMS*. IFPRI.
* Robinson, S., Cattaneo, A. & El-Said, M. (2001). Updating and estimating a SAM using cross entropy methods. *Economic Systems Research* 13(1).
* Rutherford, T. (1998). *CES preferences and technology: a practical introduction* (calibrated share form). GAMS/MPSGE documentation.
* Shoven, J. & Whalley, J. (1984). Applied general-equilibrium models of taxation and international trade. *JEL* 22(3).
* Tuerck, D., Haughton, J., Barrett, J. & Codreanu, S. (2004). *Texas-STAMP: A Sophisticated Tax Model for Texas*. Texas Public Policy Foundation.
