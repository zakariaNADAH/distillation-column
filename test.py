# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 00:13:22 2022

@author: asus
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as stc
import math as mt


st.set_page_config(page_title="Design App", layout='wide', page_icon="chart_with_upwards_trend")
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

col = st.columns(4)
with col[0]:
    st.image("Logo_UT3.jpg")
    
with col[3]:
    st.image("ensiacet.jpg")

st.markdown('---')

st.sidebar.header("Définition des paramètres")

liste = ['Colonne de distillation', 'Réacteur à lit fixe']
techno = st.radio('', liste, index=0)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


if techno == liste[0]:

    HTML_BANNER = """
        <h1 style="color:#DD985C;text-align:center;">Colonne de distillation</h1>
        <p style="color:#DD985C;text-align:center;">Méthode de MacCabe-Thiele</p>
        </div>
        """
    stc.html(HTML_BANNER)
    st.markdown("Le dimensionnement d'une colonne de distillation consiste à déterminer son diamètre et sa hauteur. Dans cet application, on va voir les étapes clés pour déterminer **la hauteur d'une colonne de distillation** par la méthode de **MacCabe-Thiele**.")
    
    
    # =============================================================================
    # Définir la fonction d'équilibre
    # =============================================================================
    with st.sidebar.form("my_form"):
        #st.write("Définition des paramètres")
        alpha = st.text_input("Volatilité relative", 2.5)
        alpha = float(alpha)
        q = st.number_input("La fraction du vapeur dans l'alimentation (q)", value= 1.0)
        q = float(q)
        if q==1:
            q=0.9999999999
        
        X_F = st.slider("Titre molaire de l'alimentation (en plus volatil XF)",min_value=0.0, max_value=100.0, step=0.1, value=40.0)
        X_F = X_F/100
        X_D = st.slider("Titre molaire du distillat (en plus volatil XD)",min_value=0.0, max_value=100.0, step=0.1, value=95.5)
        X_D = X_D/100
        X_W = st.slider("Titre molaire du résidu (en plus volatil XW)",min_value=0.0, max_value=100.0, step=0.1, value=06.0)
        X_W = X_W/100

        # Every form must have a submit button.
        submitted = st.form_submit_button("Simulate")
    
    def equi(alpha):
        x_eq = np.linspace(0, 1, 101)
        y_eq = alpha*x_eq/(1+(alpha-1)*x_eq)
        return x_eq, y_eq
    
    x_eq, y_eq = equi(alpha)
    
    
    
    st.write("**1. Le nombre minimum d'étages théoriques ($NET_{min}$ à Reflux total)**")
    # =============================================================================
    # Définir quelques paramétres
    # =============================================================================
    
    
    
    # =============================================================================
    # Déterminer Rmin : il faut définir une fonction qui nous retourne le point 
    # d'intersection entre la courbe d'alimentation et la courbe d'équilibre
    # =============================================================================
    
    def inter(q, X_F, alpha):
        c1 = (q*(alpha-1))
        c2 = q + X_F*(1-alpha) - alpha*(q-1)
        c3 = -X_F
        coeff = [c1, c2, c3]
        r = np.sort(np.roots(coeff))
        
        if r[0]>0:
            x_ae = r[0]
        else:
            x_ae = r[1]
       
        y_ae = alpha*x_ae/(1+ x_ae*(alpha-1))
        if q == 1:
            x_fed = [X_F, X_F]
            y_fed = [X_F, y_ae]
        else:
            x_fed = np.linspace(X_F, x_ae, 51)
            y_fed = q/(q-1)*x_fed - X_F/(q-1)
        
        return x_ae, y_ae, y_fed, x_fed
    x_ae, y_ae, y_fed, x_fed = inter(q, X_F, alpha)
    
    # =============================================================================
    # NET min
    # =============================================================================
    R = 1000
    
    x_inter = (X_F/(q-1)+X_D/(R+1))/(q/(q-1)-R/(R+1))
    y_inter = R/(R+1)*x_inter + X_D/(R+1)
    
    # =============================================================================
    # Section de rectification : établissement de la courbe d'enrichissement
    # =============================================================================
    
    def rect(R, X_D, x_inter):
        x_rect = np.linspace(X_D, x_inter, 51)
        y_rect = R/(R+1)*x_rect +X_D/(R+1)
        return x_rect, y_rect
    
    x_rect, y_rect = rect(R, X_D, x_inter)
    
    # =============================================================================
    # Section d'allimentation : établissement de la courbe d'allimentation
    # =============================================================================
    
    def alim(X_F, q, x_inter):
        x_alim = np.linspace(X_F, x_inter)
        y_alim = q/(q-1)*x_alim - X_F/(q-1)
        return x_alim, y_alim
    
    x_alim, y_alim = alim(X_F, q, x_inter)
    
    # =============================================================================
    # Section d'appauvrissement : établissement de la courbe d'appauvrissement
    # =============================================================================
    
    def appau(X_W, x_inter, y_inter):
        x_appau = np.linspace(X_W, x_inter, 51)
        y_appau = (y_inter - X_W)/(x_inter - X_W) * (x_appau - X_W) +X_W
        return x_appau, y_appau
    x_appau, y_appau = appau(X_W, x_inter, y_inter) 
    
    # =============================================================================
    # Construction des étages
    # =============================================================================
    s = np.zeros((1000,5)) # Empty array (s) to calculate coordinates of stages
    
    for i in range(1,1000):
        # (s[i,0],s[i,1]) = (x1,y1) --> First point
        # (s[i,2],s[i,3]) = (x2,y2) --> Second point
        # Joining (x1,y1) and (x2,y2) will result into stages
        
        s[0,0] = X_D
        s[0,1] = X_D
        s[0,2] = s[0,1]/(alpha-s[0,1]*(alpha-1))
        s[0,3] = s[0,1]
        s[0,4] = 0
    # x1
        s[i,0] = s[i-1,2]
        
        # Breaking step once (x1,y1) < (xW,xW)
        if s[i,0] < X_W:
            s[i,1] = s[i,0] 
            s[i,2] = s[i,0]
            s[i,3] = s[i,0]
            s[i,4] = i
            break
            # y1
        if s[i,0] > x_inter:
            s[i,1] = R/(R+1)*s[i,0] + X_D/(R+1)
        else :
            s[i,1] = ((y_inter - X_W)/(x_inter - X_W))*(s[i,0]-X_W) + X_W
            
        # x2
        if s[i,0] > X_W:
            s[i,2] = s[i,1]/(alpha-s[i,1]*(alpha-1))
        else:
            s[i,2] = s[i,0]
        
        # y2
        s[i,3] = s[i,1]
        
        # Nbr des étages
        if s[i,0] < x_inter:
            s[i,4] = i
        else:
            s[i,4] = 0
    
    s = s[~np.all(s == 0, axis=1)] # Clearing up zero containing rows 
    s_rows = s.shape[0] 
    
    S = np.zeros((s_rows*2,2)) # Empty array to rearragne 's' array for plotting
    
    for i in range(0,s_rows):
        S[i*2,0] = s[i,0]
        S[i*2,1] = s[i,1]
        S[i*2+1,0] = s[i,2]
        S[i*2+1,1] = s[i,3]
    
    # =============================================================================
    # Déterminier le nombre des étages théoriques
    # =============================================================================
    x_s = s[:,2:3]
    y_s = s[:,3:4]
    
    stages = np.char.mod('%d', np.linspace(1,s_rows-1,s_rows-1))
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    for label, x, y in zip(stages, x_s, y_s):
        plt.annotate(label,
                      xy=(x, y),
                      xytext=(0,5),
                      textcoords='offset points', 
                      ha='right')
    
    plt.grid(linestyle='dotted')
    #plt.title('Distillation Column Design (MacCabe-Thiele Method)')
    # if courbe_équi == liste_équi[1]:
    #     plt.scatter(xx, yy, marker='o', s=10)
    plt.plot(x_eq,y_eq,'-', label="Courbe d'équilibre")
    plt.plot([0, 1],[0, 1],'black')
    
    plt.scatter(X_D,X_D, color='r', s=20)
    plt.scatter(X_F,X_F, color='r', s=20)
    plt.scatter(X_W,X_W, color='r', s=20)
    
    plt.plot(S[:,0],S[:,1],'r-.', label="Etages")
    
    plt.legend(loc="upper left")
    plt.xlabel("x (-)")
    plt.ylabel("y (-)")
    
    st.pyplot()
    
    stages_min = s_rows -1
    #st.write("Le nombre des étages theoriques minimal pour réaliser cette séparation est:", étages_min,"étages")
    st.write(r''' $$\hspace*{5.2cm} NET_{min} =$$''', stages_min)
    if stages_min > 15:
        st.error("Attention!! le nombre des étages theoriques minimal est trop élevé. La distillation à ces conditions n'est pas raisonnable ")
    else :
        st.success("Parfait! l'opération qu'on souhaite mise en oeuvre est raisonnable")
    
    
    
    
    # =============================================================================
    # Rmin
    # =============================================================================
    st.write(r'**2. Le taux de reflux minimum ($R_{min}$ à nombre de plateaux infini)**')
    
    def Rmin(X_D, x_ae, y_ae):
        x_Rmin = np.linspace(X_D, 0, 51)
        y_Rmin = (y_ae - X_D)/(x_ae - X_D) * (x_Rmin - X_D) +X_D
        return x_Rmin, y_Rmin
    x_Rmin, y_Rmin = Rmin(X_D, x_ae, y_ae) 
    
    ######## R_min & R (new) ########
    R_min = (X_D-y_ae)/(y_ae - x_ae)
    ordo = X_D/(R_min +1)
    
    # =============================================================================
    # plot
    # =============================================================================
    
    plt.grid(visible=True, which='major',linestyle=':',alpha=0.6)
    plt.grid(visible=True, which='minor',linestyle=':',alpha=0.3)
    plt.minorticks_on()
    #plt.title('Distillation Column Design (MacCabe-Thiele Method)')
    
    plt.plot(x_eq,y_eq,'-', label="Courbe d'équilibre")
    plt.plot([0, 1],[0, 1],'black')
    
    plt.scatter(X_D, X_D, color='r', s=20)
    plt.scatter(X_F, X_F, color='r', s=20)
    plt.scatter(x_ae, y_ae, color='r', s=20)
    
    plt.plot(x_Rmin, y_Rmin, label="Courbe d'enrichissement")
    plt.plot(x_fed,y_fed, label="Courbe d'alimentation")
    
    plt.legend(loc="best")
    plt.xlabel("x (-)")
    plt.ylabel("y (-)")
    
    plt.scatter(0,ordo, color='r', s=20)
    plt.text(0.01,ordo-0.08,'($\\frac{X_{D}}{R_{min}+1}$)',horizontalalignment='center')
    st.pyplot()
    
    st.markdown(r'### <p style="text-align: center;">$$R_{min}=\frac{X_{D}}{Y_{min}}-1$$</p>', unsafe_allow_html=True)
    st.write("$\hspace*{5.2cm} R_{min} =$",round(R_min,3))
    
    
    st.write("**3. Le nombre d'étages théoriques NET requis pour un taux de reflux R :**")
    col1 = st.columns(2)
    
    with col1[1]:
        Coeff = st.slider("Coeff",min_value=1.0, max_value=2.0, step=0.01, value=1.21)
        R = Coeff*R_min
    
    with col1[0]:
        st.write("Le taux de reflux réel $R$ est définit par :")
        st.write("$R = Coeff \\times R_{min}$")
        st.write("$R =$",round(R,3))
    
    
    # =============================================================================
    # le point d'intersection entre la courbe d'alimentation et la courbe d'enrichissement
    # =============================================================================
    
    x_inter = (X_F/(q-1)+X_D/(R+1))/(q/(q-1)-R/(R+1))
    y_inter = R/(R+1)*x_inter + X_D/(R+1)
    
    # =============================================================================
    # Section de rectification : établissement de la courbe d'enrichissement
    # =============================================================================
    
    def rect(R, X_D, x_inter):
        x_rect = np.linspace(X_D, x_inter, 51)
        y_rect = R/(R+1)*x_rect +X_D/(R+1)
        return x_rect, y_rect
    
    x_rect, y_rect = rect(R, X_D, x_inter)
    
    # =============================================================================
    # Section d'allimentation : établissement de la courbe d'allimentation
    # =============================================================================
    
    def alim(X_F, q, x_inter):
        x_alim = np.linspace(X_F, x_inter)
        y_alim = q/(q-1)*x_alim - X_F/(q-1)
        return x_alim, y_alim
    
    x_alim, y_alim = alim(X_F, q, x_inter)
    
    # =============================================================================
    # Section d'appauvrissement : établissement de la courbe d'appauvrissement
    # =============================================================================
    
    def appau(X_W, x_inter, y_inter):
        x_appau = np.linspace(X_W, x_inter, 51)
        y_appau = (y_inter - X_W)/(x_inter - X_W) * (x_appau - X_W) +X_W
        return x_appau, y_appau
    x_appau, y_appau = appau(X_W, x_inter, y_inter) 
    
    # =============================================================================
    # Construction des étages
    # =============================================================================
    s = np.zeros((1000,5)) # Empty array (s) to calculate coordinates of stages
    
    for i in range(1,1000):
        # (s[i,0],s[i,1]) = (x1,y1) --> First point
        # (s[i,2],s[i,3]) = (x2,y2) --> Second point
        # Joining (x1,y1) and (x2,y2) will result into stages
        
        s[0,0] = X_D
        s[0,1] = X_D
        s[0,2] = s[0,1]/(alpha-s[0,1]*(alpha-1))
        s[0,3] = s[0,1]
        s[0,4] = 0
    # x1
        s[i,0] = s[i-1,2]
        
        # Breaking step once (x1,y1) < (xW,xW)
        if s[i,0] < X_W:
            s[i,1] = s[i,0] 
            s[i,2] = s[i,0]
            s[i,3] = s[i,0]
            s[i,4] = i
            break
            # y1
        if s[i,0] > x_inter:
            s[i,1] = R/(R+1)*s[i,0] + X_D/(R+1)
        else :
            s[i,1] = ((y_inter - X_W)/(x_inter - X_W))*(s[i,0]-X_W) + X_W
            
        # x2
        if s[i,0] > X_W:
            s[i,2] = s[i,1]/(alpha-s[i,1]*(alpha-1))
        else:
            s[i,2] = s[i,0]
        
        # y2
        s[i,3] = s[i,1]
        
        # Nbr des étages
        if s[i,0] < x_inter:
            s[i,4] = i
        else:
            s[i,4] = 0
    
    s = s[~np.all(s == 0, axis=1)] # Clearing up zero containing rows 
    s_rows = s.shape[0] 
    
    S = np.zeros((s_rows*2,2)) # Empty array to rearragne 's' array for plotting
    
    for i in range(0,s_rows):
        S[i*2,0] = s[i,0]
        S[i*2,1] = s[i,1]
        S[i*2+1,0] = s[i,2]
        S[i*2+1,1] = s[i,3]
    
    # =============================================================================
    # Déterminier le nombre des étages théoriques
    # =============================================================================
    # (x2,y2) from 's' array as (x_s,y_s) used for stage numbering
    x_s = s[:,2:3]
    y_s = s[:,3:4]
    
    stages = np.char.mod('%d', np.linspace(1,s_rows-1,s_rows-1))
    
    NET = s_rows-1
    
    # '''
    # localiser l'étage d'alimentation
    # '''
    s_f = s_rows-np.count_nonzero(s[:,4:5], axis=0)
    
    # =============================================================================
    # FINALE
    # =============================================================================
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    
    fig = plt.figure(num=None, figsize=(10, 8))
    
    for label, x, y in zip(stages, x_s, y_s):
        plt.annotate(label,
                      xy=(x, y),
                      xytext=(0,5),
                      textcoords='offset points', 
                      ha='right')
    
    plt.grid(linestyle='dotted')
    plt.title('Distillation Column Design (MacCabe-Thiele Method)')
    
    plt.plot(x_eq,y_eq,'-', label="Courbe d'équilibre")
    plt.plot([0, 1],[0, 1],'black')
    
    
    plt.scatter(X_D,X_D, color='r' )
    plt.scatter(X_F,X_F, color='r' )
    plt.scatter(X_W,X_W, color='r' )
    
    plt.scatter(x_inter,y_inter )
    plt.plot(x_alim, y_alim, label="Courbe d'alimentation")
    plt.plot(x_appau, y_appau, label="Courbe d'appauvrissement")
    plt.plot(x_rect, y_rect, label="Courbe d'enrichissement")
    # plt.plot(x_fed,y_fed, color='black' )
    
    plt.plot(S[:,0],S[:,1],'-.', label="Etages")
    
    plt.legend(loc="upper left")
    plt.xlabel("x (-)")
    plt.ylabel("y (-)")
    
    st.pyplot()
    
    st.write(r''' $$\hspace*{5.2cm} NET =$$''', s_rows -1)
    
    st.write("**4. Hauteur de la colonne**")
    
    menu = ["Colonne à plateaux","Colonne à garnissage"]
    techno = st.selectbox("Technologies",menu)
    
    st.markdown("La hauteur de la colonne résulte:")
    st.write("$~~~~$- du nombre d'étages théoriques nécessaires")
    
    if techno == menu[0]:
        st.write("$~~~~$- de l'efficacité de chaque plateau réel (eff)")
        st.write("$~~~~$- de l'espacement entre plateaux (TS pour Tray Spacing)")
        st.markdown(r'### <p style="text-align: center;">$$H=\frac{NET}{eff} \times TS$$</p>', unsafe_allow_html=True)
        
        col2 = st.columns(2)
        with col2[0]:
            eff = st.text_input("efficacité des plateaux (%) ", 90)
            eff = float(eff)
        
        with col2[1]:
            TS = st.text_input("espacement entre plateaux (m) ", 0.4)
            TS = float(TS)
        st.write("- Hauteur de la colonne =", round(NET/(eff/100)*TS,2),"m")
    else:
        st.write("$~~~~$- de la hauteur equivalente à un plateau théorique (HEPT)")
        st.markdown(r'### <p style="text-align: center;">$$H=NET \times HEPT$$</p>', unsafe_allow_html=True)
        col2 = st.columns(2)
        with col2[0]:
            HEPT = st.text_input("Hauteur Equivalente à un Plateau Théorique (m)", 0.8)
            HEPT = float(HEPT)
        st.write("- Hauteur de garnissage à installer =",round(NET*HEPT,2),"m" )


else:
    HTML_BANNER = """
        <h1 style="color:#DD985C;text-align:center;">Réacteur catalytique à lit fixe</h1>
        <p style="color:#DD985C;text-align:center;">Les étapes-clés de dimensionnement</p>
        </div>
        """
    stc.html(HTML_BANNER)
    
    #st.title("Stratégie de dimensionnement d'un réacteur catalytique à lit fixe : les étapes-clés")
    st.subheader('Description de la problèmatique : Réacteur de production de formaldéhyde')
    st.markdown("Concevoir un réacteur d'oxydation du méthanol en formaldéhyde sur catalyseur Fe-Mo en réacteur à lit fixe, en considérant que c'est une réaction du premier ordre :")
    st.latex(r''' CH_3OH ~~ + ~~  1/2 ~ O_2 ~~~ \to ~~~ CH_2O ~~  +  ~~ H_2O''')
    
    k_s = 5e-04
    st.write("$r_s$ = $k_s$ $\\times$ $C_s$ $~~$ avec $~~$ $k_s$ =",k_s,"cm/s $~~$ à $~~$ 300°C")
    
    st.markdown("Dans ce travail les calculs de pré-dimensionnement sont effectués en considérant d'abord le réacteur isotherme (pour simplifier les calculs).")
    
    st.subheader("Objectifs")
    cols = st.columns(2)
    with cols[0]:
        Q_p = st.text_input("Production visé de CH2O en Kg/h", 1000)
        
    with cols[1]:
        X = st.slider("Le taux de conversion", min_value=0.0, max_value=1.0, value=0.99)
    Q_p = float(Q_p)
    X = float(X)
    
# =============================================================================
#     set sidebar
# =============================================================================
    st.sidebar.markdown("**Conditions opératoires**")
    temp = st.sidebar.slider('Température (°C)',min_value=0.0, max_value=500.0, step=1.0, value=300.0)
    Pression = st.sidebar.slider('Pression (atm)', 0, 20, 1)
    frac = st.sidebar.slider('Fraction molaire du réactif clé (-)',min_value=0.0, max_value=1.0, step=0.01, value=0.03)
    Epsi_lit = st.sidebar.slider("Porosité du lit (-)", min_value=0.0, max_value=1.0, step=0.01, value=0.4)
    
    
    st.sidebar.markdown("**Données cinétiques**")
    Ea = st.sidebar.text_input("Energie d'activation (cal/mol)", value=19000)
    Ea = float(Ea)
    #delta_h = st.sidebar.number_input("delta_h (cal/mol)", value=-38500)
    delta_h = -38500
    
    st.sidebar.markdown("**Propriétés du catalyseur**")
    #delta_grain = st.sidebar.number_input('diamètre de la couche limite autour du grain en A°', value=50)
    delta_grain = 50
    rho_grain = st.sidebar.text_input('Masse volumique(Kg/cm3)',1.1)
    rho_grain = float(rho_grain)
    porosity_grain = st.sidebar.slider('Porosité du grain (-)', min_value=0.0, max_value=1.0, step=0.01, value=0.66)
    tau_grain = st.sidebar.text_input('Tortuosité du grain (-)', value=4)
    tau_grain = float(tau_grain)
    #Si = st.sidebar.text_input('Si',1e05)
    Si = 1e05
    #lambda_ef = st.sidebar.text_input('lambda_ef (cal/cm/s/K)',5e-04)
    lambda_ef = 5e-04
    
    st.sidebar.markdown("**Propriétés du fluide**")
    rho_gaz = st.sidebar.text_input('Masse volumique du gaz (Kg/m3)',0.62)
    rho_gaz = float(rho_gaz)
    Cp = st.sidebar.text_input('Capacité calorifique du gaz (cal/g/K)', value=0.25)
    Cp = float(Cp)
    #D_m = st.sidebar.text_input('diffusivité du gaz dans le mélange (m2/s)', 5e-05)
    D_m = 5e-05
    #mu_gaz = st.sidebar.text_input('mobilité du gaz (Pa.s)', 0.25e-04)
    mu_gaz = 0.25e-04
    lambda_gaz = 1.1e-04
    M_i = 32
    #M_i = st.sidebar.number_input('masse molaire du réactif clé en g/mol', value=32)
    
    
    st.subheader("Les différents paramètres")
    para = st.columns(3)
    with para[0]:
        st.markdown("**Opératoire**")
        st.write('T =', temp, '°C')
        st.write('P =', Pression, 'atm')
        st.write("$x_i$=", frac*100, '%')
        st.write("$\\varepsilon_{lit}$=",Epsi_lit)
        
    with para[1]:
        st.markdown("**Catalyseur**")
        st.write("$\\rho_p$=", rho_grain, 'g/cm3')
        st.write("$\\varepsilon_{p}$=", porosity_grain)
        st.write("$\\tau_p$=", tau_grain)
        st.write("$\\delta_p$=", delta_grain,"A°")
        st.write("$Si_p$=", Si,"$cm^2$.$g^{-1}$")
        st.write('$\\lambda_{ef}$ =', lambda_ef,"cal/(cm.s.K)")
        
    with para[2]:
        st.markdown("**Gaz**")
        st.write('$D_m$ =', D_m,"$m^2$.$s^{-1}$")
        st.write("$\\rho_{gaz}$=", rho_gaz, '$kg.m^{-3}$')
        st.write("$C_p$=", Cp,"$cal.g^{-1}.K^{-1}$")
        st.write('$\\mu_{gaz}$ =', mu_gaz,"$Pa.s$")
        
    
    st.markdown('---')
    
    st.subheader("I. Identification du réactif limitant")
    st.write("A ce stade, on doit, en toute rigueur, identifier l'espèce réactive dont la migration diffusionnelle dans le grain de catalyseur sera limitante, c'est-à-dire (si on se réfère au dénominateur du module de Thiele et en considérant que les coefficient $D_{m,i}$ et $D_{e,i}$ varient dans les mêmes proportions pour les différentes espèces) celle dont le terme $\\frac{D_{m,i}.C_i}{v_i}$ sera le plus faible.")
    st.write("Dans l'exemple traité, il faudrait donc connaître les coefficients de diffusion moléculaires dans l'air du méthanol et de l'oxygène.")
    st.write("Mais ici, le méthanol est présent en faible concentration dans le courant gazeux alimentant le réacteur ; de plus, la taille de la molécule de méthanol est grande devant celle de l'oxygène, et il est consommé deux fois plus vite que l'oxygène par la réaction considérée. On admet donc implicitement que le méthanol est l'espèce qui diffuse le moins rapidement dans le matériau catalytique, et qu'il sera donc l'espèce limitante pour la réaction, dans le grain.")
    
    
    st.subheader("II. Clacul de la taille des grains de catalyseur", anchor='')
    '''
    Dans un premier temps on suppose négligeables les gradients de concentration et 
    de température dans la couche limite autour des grains de catalyseur (résistance
    externe aux transferts négligeable), donc:
    '''
    st.latex('''C_s = C_e ~~ et ~~  T_s = T_e''')
    
    st.markdown('- ***Coefficient de diffusion dans les grains*** :')
    st.write(r''' $$\frac{1}{D_p} = \frac{1}{D_m} + \frac{1}{D_K} ~~~~ avec ~~ D_K = \frac{1}{3}\delta_p~ \sqrt[]{\frac{8RT}{\pi M}} = 1.534~\delta_p~ \sqrt[]{\frac{T}{M}} ~~~ et ~~~ D_{eff} = \frac{\varepsilon_p D_p}{\tau_p} ~~~(S.I)$$''')
    
    D_K = 1.534 * delta_grain * 1e-10 * np.sqrt((temp+273.15)/(M_i*1e-03))
    D_p = 1/((1/D_m) + (1/D_K))
    D_eff = porosity_grain * D_p/tau_grain
    
    D = st.columns(3)
    with D[0]:
        st.write("$D_K$ = ",round(D_K*1e06,2),'$10^{-06}~m^2$/s')
        
    with D[1]:
        st.write("$D_p$ = ",round(D_p*1e06,2),'$10^{-06}~m^2$/s')
        
    with  D[2]:
        st.write("$D_{eff}$ = ",round(D_eff*1e07,2),'$10^{-07} ~~m^2$/s')
    
    
    st.markdown('- ***Calcul du module de Thiele $\Phi_s$*** :')
    st.write("Conversion en unité S.I. de la constante cinétique à",temp,"°C")
    st.latex(r''' k_i = k_s \times S_i \times \rho_p''')
    k_i = k_s * Si * rho_grain
    st.write('$k_i$ = ',round(k_i, 3),'$s^{-1}$')
    
    Remarque_1 = st.expander('Voir plus')
    with Remarque_1:
        st.caption("Dans le domaine de la catalyse hétérogène, une constante de vitesse exprimée en $s^{-1}$ ou $min^{-1}$ donne généralement une vitesse de réaction en mol/$m^3_{cata}$/s (si les scientifiques qui ont identifié la cinétique l'ont bien exprimée par unité de volume de catalyseur et non de réacteur)")
    
    
    st.latex(r''' en~~régime~~chimique~~ \Phi_s = \frac{d_p}{6} ~~\sqrt[]{\frac{k_i}{D_eff}} = 0.3''')
    
    Phi_s = 0.3
    d_p = 6*Phi_s*(1/np.sqrt(k_i/D_eff))
    
    st.write("$d_p$ = ",round(d_p*1e06,2),"$10^{-06}~ m$")
    
    Remarque_2 = st.expander('Voir plus')
    with Remarque_2:
        if d_p < 0.001:
            st.caption("Cette taille de particule est trop faible : les pertes de charges dans un réacteur rempli de ces grains seraient trop élevées.")
        else :
            st.caption('Done')
       
    if d_p < 0.001:
        st.markdown("On choisit un catalyseur de type **coquille d'œuf**, de diamètre **1 mm** (valeur de départ). Ce diamètre comprene une couche de catalyseur d'épaisseur **e**")
        cols1 = st.columns(2)
        with cols1[0]:
            d_p = st.slider('Le diamètre des particules en mm',min_value=0.0, max_value=10.0, step=0.1, value=1.0)
        with cols1[1]:
            e = st.number_input("e : l'épaisseur de la couche de catalyseur en µm", value=50)
            
        R = (d_p/2) *1e-03
        V_particule = 4/3 * np.pi * (R)**3
        # cols2 = st.columns(2)
        # with cols2[0]:
        #     e = st.number_input("Choisir l'épaisseur de la couche de catalyseur en µm", value=50)
    
    
        
    st.markdown("- ***Calcul du module de Thiele $\Phi_s$ et de l'efficacité $\eta_s$ exacts pour ces grains*** :")   
    st.latex(r''' \Phi_s = e ~~\sqrt[]{\frac{k_i}{D_eff}} = 0.3''')
      
    Phi_s =  e*1e-6 * np.sqrt(k_i/D_eff)
    eta_s = mt.tanh(Phi_s)/Phi_s   
    
    
    st.write('$\Phi_s$ = ', round(Phi_s,3))
    st.write("$\eta_s$ = ", round(eta_s,3))
    
    if Phi_s < 0.3:
        st.success("Le réacteur fonctionne en **régime chimique**")
        
    elif Phi_s >3:
        st.error("Le réacteur fonctionne en **régime diffusionnel**")
    
    else:
        st.warning("Le réacteur fonctionne en **régime intermédiaire**")
    
    st.subheader("III. Dimensionnement d'un réacteur isotherme")
    st.markdown("- ***Conservation du débit total de gaz***")
    
    Q_r = Q_p/3600/30e-03/X*M_i*1e-03/rho_gaz/frac
    st.write("Le débit volumique total de gaz, $Q_v$ =", round(Q_r,3), "$m^3$/s")
    
    Q = Q_r
    st.markdown("- ***Volume du réacteur isotherme***")
    
    st.markdown("On peut alors appliquer la relation (valable pour une réaction d'ordre 1) :")
    st.latex(r'''V_{cata} = \frac{-Q}{\eta_s k_i} ~\ln{(1-X)}''')
    
    V_cata = -Q/(eta_s*k_i)*mt.log(1-X)
    st.write("On trouve $V_{cata}$ =", round(V_cata,2),"m3")
    
    st.write("Ce volume représente le volume de catalyseur nécessaire pour réaliser ",X*100,"% de conversion.")
    st.write("Ce volume est uniquement constitué des couronnes de support qui ont été imprégnées de substances actives sur chaque particule (coquilles).")
    st.write("On calcule alors le volume occupé par les billes, puis le volume du cylindre qui contient le lit de billes :")
    
    st.latex(r''' V_{cata~~dans~~grain} = \frac{4}{3} \pi R^3 - \frac{4}{3} \pi (R-e)^3''')
    
    Epsi_coq = ((4/3 * np.pi * (R)**3)-(4/3 * np.pi * (R-e*1e-06)**3)) / V_particule
    st.latex(r''' \frac {V_{cata~~dans~~grain}}{V_{particule}} = \frac {R^3 -(R-e)^3}{R^3} = \varepsilon_{coquille} ''')
    
    V_reacteur = V_cata / (Epsi_coq * (1 - Epsi_lit))
    st.latex(r''' Ainsi : V_{cata} = V_{réacteur} (1-\varepsilon_{lit}).\varepsilon_{coquille} ~~~ et~~ donc : ~~~V_{réacteur} = \frac{V_{cata}}{(1-\varepsilon_{lit}).\varepsilon_{coquille}} ''')
    
    st.write("$V_{réacteur}$ =", round(V_reacteur,2),"$m^3$")
    st.write("$$\\varepsilon_{coquille}$$ =", round(Epsi_coq,3),"$m^3$")
    
    
    st.subheader("IV. Analyse des transferts externes")
    st.markdown("- **Choix d'une vitesse (superficielle) $v_z$ de circulation des gaz**")
    
    cols3 = st.columns(2)
    with cols3[0]:
        v_z = st.slider('La vitesse superficielle en m/s',min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    
    st.markdown("Pour cette vitesse la section droite du réacteur est :")
    st.latex(r''' S_{réc} = \frac{Q}{v_z}''')
    
    Section_rea = Q/v_z
    st.write("$S_{réc} =$ ",round(Section_rea,2),'$m^2$')
    st.write("Les dimensions du réacteur, respectant $V_{réacteur}=$",round(V_reacteur,2),"$m^3$, seraient donc  :")
    
    D_t = 2*mt.sqrt(Section_rea/mt.pi)
    L_R = V_reacteur/Section_rea
    
    st.write("$D_t =$",round(D_t,3), "m $~~$  et $~~$  $L_R =$",round(L_R,3), "m  $~~$ ce qui donne $~~$ $L_R$/$D_t =$", round(L_R/D_t,2))
    
    
    if 1 < L_R/D_t <20:
        st.success("Parfait : Ceci respecte 1< $L_R$/$D_t$ < 20")
    
    else :
        st.error("Le réacteur est beaucoup plus court devant son diamètre le critère 1< $L_R$/$D_t$ < 20 n'est plus vérifié. ")
        st.info("Suggestion (augmenter la vitesse $v_z$ ou augmenter le diamètre des particules (sans modifier la valeur de **e**), pris initialement à **1 mm** par défaut).")
    
    st.markdown("- **Estimation du gradient de masse près du grain (face externe)**")
    
    st.markdown("Calcul de $k_D$ :")    
    st.latex(r''' S_h = \frac{k_D.d_p}{D_m} = 2 + 1.8~~. R_e^{1/2} .S_c^{1/3} ~~~~~~avec~~R_e = \frac{\rho . u. d_p}{\mu}~~~et~~~ S_c = \frac{\mu}{\rho . D}''')
    
    Re = rho_gaz * v_z * R*2 / (mu_gaz )
    Sc = mu_gaz/rho_gaz/D_m
    st.write("On estime $R_e$ =",round(Re,3), "et $S_c$ =",round(Sc,3))
    
    Sh = 2 + 1.8 * Re**(1/2) * Sc**(1/3)
    k_D = D_m * Sh / (R*2)
    st.write("Alors on trouve $S_h$ =",round(Sh, 3), "et $k_D$ =",round(k_D,3),'m/s')
    
    st.markdown("Le gradient massique externe est alors donné par :")
    st.latex(r''' C_e - C_s = \frac{r_{app}.L}{k_D} = \frac{\eta_s .k_i .C_s .e}{k_D},~~~en~~~prenant~~~ici~~~C_s \simeq C_A ^e = x_A \times \frac{P}{RT}''')
    
    Cs = frac * Pression * 1.01325e05 / 8.32/(temp+273.15)
    st.write("$C_s$ =", round(Cs,3),"mol/L")
    
    C_e_s = eta_s * k_i * Cs * e*1e-06 / k_D
    
    st.write("On trouve $\\frac{C_e - C_s}{C_s}$ =",round(C_e_s/Cs* 100,2) ,"%. Ce gradient est négligeable.")
    
    st.markdown("- **Estimation du gradient thermique près du grain**")
    st.markdown("Calcul de h :")    
    st.latex(r''' N_u = \frac{h.d_p}{\lambda} = 2 + 1.8~~. R_e^{1/2} .P_r^{1/3} ~~~~~~avec~~ P_r = \frac{C_{pm}.\mu}{\lambda}''')
    
    st.write("On a toujours $R_e$ =",round(Re,3))
    
    Pr = Cp * 4.18 * 1000 * mu_gaz / (lambda_gaz * 100 * 4.18)
    Nu = 2 + 1.8*Re**0.5 * Pr**(1/3)
    h = Nu * lambda_gaz * 100 / (d_p*1e-03)
    st.write("On estime $P_r$ =", round(Pr,3), "$~~$ alors $N_u$ =",round(Nu,3), " $~~$ et $~~$ $h$ =",round(h,3),"W/m/K")
    st.write("Le gradient thermique externe est alors donné par : $T_e -T_s$ = $\\frac{r_{app}.L.\Delta H_R}{h}$ = $\\frac{\eta_s .k_i .C_s .e.\Delta H_R}{h}$ ")
    st.write(" avec $~~T_e$ =",temp +273 ,"K")
    
    Ts = temp+273 - eta_s * k_i * Cs * e*1e-06 * delta_h / h
    st.write("On trouve $T_s$ =",round(Ts,2),"K, ce qui est très proche de $T_e$ (",round(Ts-(temp+273),3),"K d'écart : insignifiant devant $T_e$ =",temp +273,"K).")
    
    if Ts-(temp+273) < (temp+273)*0.1 :
        if  C_e_s/Cs* 100 < 10:
            st.success("Pour $d_p$ et $v_z$ choisis, Les transferts externes ne sont pas limitants pour la réaction chimique.")
        #st.write(perte_charge*100)
    
    
    
    st.subheader("V. Évaluation des pertes de charge")
    
    st.latex(r''' -\frac{\Delta P}{\Delta z} = \frac{f .\rho_G .v_z^2}{d_p} ~~~~ avec ~~~~ f = \frac{1-\varepsilon_{lit}}{\varepsilon_{lit}^3} .(1.8 + 180 .\frac{1-\varepsilon_{lit}}{Re_p})''')
    
    f = (1-Epsi_lit)/Epsi_lit**3 * (1.8 + 180*(1-Epsi_lit)/Re)
    perte = f * rho_gaz * v_z**2/ (2*R)
    perte_charge = perte*L_R/(Pression*1.01325e05)
    st.write("On trouve f =",round(f,2), "et   -$\\frac{\Delta P}{\Delta z}$  =", round(perte,2), "$Pa.m^{-1}$. Cela donne $\\frac{\Delta P}{P}$ =", round(perte_charge*100, 2),"%")
    
    
    if perte_charge*100 >30:
        st.error("Attention : les pertes de charges sont trop élevées que la pression opératoire (plus de 30%), ce qui est irréaliste.")
        st.info("Suggestion (Diminuer la vitesse superficielle $v_z$ ou augmenter le diamètre des particules (sans modifier la valeur de **e**), pris initialement à **1 mm** par défaut).")
    
    else :
        st.success('Parfait! Les pertes de charges sont acceptable (mois de 30%)')
    
    
    st.subheader("V. Récapitulatif des résultats")
    d_p = str(d_p)
    
    df = pd.DataFrame(
        np.array([[ d_p,  int(e) , round(V_cata,2), round(V_reacteur,2), round(v_z,2), round(L_R,2), round(D_t,2), round(Re,2)]]),
        columns=('d_p (mm)','e (µm)','Vcata (m3)','Vréac (m3)','Vz (m/s)','L_R (m)','D_t (m)','Re'))
    
    st.table(df)
    
    # Ts = round(Ts, 2) 
    # Ts = str(Ts)
    
    # dg = pd.DataFrame(
    #     np.array([[ Ts,  round(perte_charge*100, 2)]]),
    #     columns=('T_s (K)','Pertes de charge (%)'))
    
    # st.table(dg)
    


Button = st.expander("Get In Touch With Me!")
with Button:
    col31, col32, col33 = st.columns(3)
    col31.write("[Zakaria NADAH](https://www.linkedin.com/in/zakaria-nadah-00ab81160/)")
    col31.write("Ingénieur Procédés Junior")
    col31.write("+336.28.80.13.40")
    col31.write("zakariaenadah@gmail.com")
    
    col33.image("profil_.jpg")