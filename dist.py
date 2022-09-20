# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 05:38:11 2022

@author: asus
"""

import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as stc
import math as mt
from scipy.optimize import fsolve

st.set_page_config(page_title="Dist App", page_icon="chart_with_upwards_trend")
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# col = st.columns(4)
# with col[0]:
#     st.image("Logo_UT3.jpg")
    
# with col[3]:
#     st.image("ensiacet.jpg")

# st.markdown('---')

    
# horizontal Menu
selected = option_menu(None, ["Home", "Design & Modeling", "Simulation"], 
    icons=['house', 'graph-up', 'caret-right-square-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

with st.sidebar:
    sidebar = option_menu(None, ['Settings'], 
        icons=['sliders'], menu_icon="cast", default_index=0)
    
    M_ = [78.11, 92.14, 46.07, 18.01528, 60.0952]
    
    options = ['Benzene', 'Toluene', 'Ethanol', 'Water', 'Propanol']
    option_1 = st.selectbox(
    'Light Key Compound (LK)',options, index=0)
    if option_1 == options[0]:
        M_LK = M_[0]
    elif option_1 == options[1]:
        M_LK = M_[1]
    elif option_1 == options[2]:
        M_LK = M_[2]
    elif option_1 == options[3]:
        M_LK = M_[3]
    elif option_1 == options[4]:
        M_LK = M_[4]
    
    option_2 = st.selectbox(
    'Heavy Key Compound (HK)',
    options, index=1)
    if option_2 == options[0]:
        M_HK = M_[0]
    elif option_2 == options[1]:
        M_HK = M_[1]
    elif option_2 == options[2]:
        M_HK = M_[2]
    elif option_2 == options[3]:
        M_HK = M_[3]
    elif option_2 == options[4]:
        M_HK = M_[4]
    
    alpha = st.text_input("Relative volatility (α)", 2.354)
    alpha = float(alpha)
    q = st.number_input("Fraction of liquid in the feed stream (q)", value= 1.0)
    q = float(q)
    if q==1:
        q=0.9999999999
    F = st.text_input("Feed (Kmol/h)", 10.0)
    F = float(F)
    
    liste = ['Mole', 'Mass']
    Frac = st.radio('Fraction', liste, index=0)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    if Frac == 'Mole':
        X_F = st.slider("LK Mole Fraction in Feed (XF)",min_value=0.0, max_value=100.0, step=0.1, value=40.0)
        X_F = X_F/100
        X_F_mass = X_F*M_LK/(X_F*M_LK + (1-X_F)*M_HK)
        
        X_D = st.slider("LK Mole Fraction in Distillate (XD)",min_value=0.0, max_value=100.0, step=0.1, value=99.2)
        X_D = X_D/100
        X_D_mass = X_D*M_LK/(X_D*M_LK + (1-X_D)*M_HK)
        
        X_W = st.slider("LK Mole Fraction in Bottoms (XB)",min_value=0.0, max_value=100.0, step=0.1, value=01.4)
        X_W = X_W/100
        X_W_mass = X_W*M_LK/(X_W*M_LK + (1-X_W)*M_HK)
        
    else:
        X_F_mass = st.slider("LK Mass Fraction in Feed (XF)",min_value=0.0, max_value=100.0, step=0.1, value=36.108542899408286)
        X_F_mass = X_F_mass/100
        X_F = X_F_mass/M_LK/(X_F_mass/M_LK + (1-X_F_mass)/M_HK)
        
        X_D_mass = st.slider("LK Mass Fraction in Distillate (XD)",min_value=0.0, max_value=100.0, step=0.1, value=99.05765930507743)
        X_D_mass = X_D_mass/100
        X_D = X_D_mass/M_LK/(X_D_mass/M_LK + (1-X_D_mass)/M_HK)
        
        X_W_mass = st.slider("LK Mass Fraction in Bottoms (XB)",min_value=0.0, max_value=100.0, step=0.1, value=01.1893598226216556)
        X_W_mass = X_W_mass/100
        X_W = X_W_mass/M_LK/(X_W_mass/M_LK + (1-X_W_mass)/M_HK)
        
    F_mass = X_F*F*M_LK + (1-X_F)*F*M_HK
    
    st.write("----")

# if sidebar == 'Settings':
        
#     with st.sidebar.form("my_form"):
#         #st.write("Définition des paramètres")
#         M_ben = 78.11
#         M_tol = 92.14
#         alpha = st.text_input("Relative volatility (α)", 2.354)
#         alpha = float(alpha)
#         q = st.number_input("Fraction of liquid in the feed stream (q)", value= 1.0)
#         q = float(q)
#         if q==1:
#             q=0.9999999999
#         F = st.text_input("Feed (Kmol/h)", 10.0)
#         F = float(F)
        
        # liste = ['Mole', 'Mass']
        # Frac = st.radio('Fraction', liste, index=0)
        # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        
        # if Frac == 'Mole':
        #     X_F = st.slider("Fraction of the most volatile component in the feed (XF)",min_value=0.0, max_value=100.0, step=0.1, value=40.0)
        #     X_F = X_F/100
        #     X_F_mass = X_F*M_ben/(X_F*M_ben + (1-X_F)*M_tol)
            
        #     X_D = st.slider("Fraction of the most volatile component in the distillate (XD)",min_value=0.0, max_value=100.0, step=0.1, value=99.2)
        #     X_D = X_D/100
        #     X_D_mass = X_D*M_ben/(X_D*M_ben + (1-X_D)*M_tol)
            
        #     X_W = st.slider("Fraction of the most volatile component in the bottom (XB)",min_value=0.0, max_value=100.0, step=0.1, value=01.4)
        #     X_W = X_W/100
        #     X_W_mass = X_W*M_ben/(X_W*M_ben + (1-X_W)*M_tol)
            
        # else:
        #     X_F_mass = st.slider("Fraction of the most volatile component in the feed (XF)",min_value=0.0, max_value=100.0, step=0.1, value=36.108542899408286)
        #     X_F_mass = X_F_mass/100
        #     X_F = 0.4
            
        #     X_D_mass = st.slider("Fraction of the most volatile component in the distillate (XD)",min_value=0.0, max_value=100.0, step=0.1, value=99.05765930507743)
        #     X_D_mass = X_D_mass/100
        #     X_D = 0.992
            
        #     X_W_mass = st.slider("Fraction of the most volatile component in the bottom (XB)",min_value=0.0, max_value=100.0, step=0.1, value=01.1893598226216556)
        #     X_W_mass = X_W_mass/100
        #     X_W = 0.014
        
        # F_mass = X_F*F*M_ben + (1-X_F)*F*M_tol

        # # Every form must have a submit button.
        # submitted = st.form_submit_button("Run")


# =============================================================================
#     Home
# =============================================================================
if selected == "Home":
    HTML_BANNER = """
        <h1 style="color:#FF4B4B;text-align:center;">Distillation Column</h1>
        <p style="color:#FF4B4B;text-align:center;">Separation of binary component</p>
        </div>
        """
    #DD985C
    stc.html(HTML_BANNER)
    st.markdown("Distillation is a process in which a liquid mixture of volatile components is separated by imparting energy to it in consideration with the boiling points of the components so that selective vaporization takes place.")
    st.image("collage1.png")
    
    st.markdown("This application presents a model of a distillation unit. Such an application can be considered as a support tool for understanding the process behavior. By giving the possibility of easily studying the sensitivity to some operating parameters (Relative volatility, Fraction of liquid in the feed, Mole fraction of the more volatile component, etc.), it provides essential information for the pre-dimensioning of such a unit.")
    #st.markdown("This version of the application is based on This application will give an example of a Fractional Distillation process. We will present this process in a progressive way. First, we will design a distillation column by computing its height and its diameter. Then, we will give particular attention to the modeling distillation process by using material and energy balance. Finally, we will compare results obtained by our application with simulation prediction obtained with Aspen Plus and DWSIM.")
    
    st.subheader("Problem")
    st.write("A feed stream flow rate of 10 kmol/h of a saturated liquid consists of 40 mol% benzene (B) and 60% toluene (T). The desired distillate composition is 99.2 mol% of benzene and a bottom product composition with 98.6 mol% of toluene.")
    st.write("The relative volatility, benzene/toluene (αBT), is 2.354. The reflux is a saturated liquid, and the column has a total condenser and a partial reboiler. The feed is at 95°C and 1 bar.")
    
    
# =============================================================================
#    Conception
# =============================================================================
elif selected == "Design & Modeling":
    HTML_BANNER = """
        <h1 style="color:#FF4B4B;text-align:center;">Distillation Column Disgn</h1>
        <p style="color:#FF4B4B;text-align:center;">Shortcut Distillation and MacCabe-Thiele Method</p>
        </div>
        """
    stc.html(HTML_BANNER)
    # st.markdown("Le dimensionnement d'une colonne de distillation consiste à déterminer son diamètre et sa hauteur. Dans cet application, on va voir les étapes clés pour déterminer **la hauteur d'une colonne de distillation** par la méthode de **MacCabe-Thiele**.")
    st.markdown("In this section, we will get some insights into how distillation columns are designed. It is essential to know at least these factors in any distillation column design:")
    
    st.markdown("- The minimum number of plates required for the separation if no product or practically no product exits the column (the condition of total reflux).")
    st.markdown("- The minimum reflux that needed to accomplish the design separation")
    st.markdown("- The actual number of trays N (or equilibrium number of trays)")
    st.markdown("- The optimum feed tray location")
    
    st.markdown("To determine all these factors, we will use two methods: **Shortcut Distillation Method**$$^*$$ and **The MacCabe-Thiele Method**.")
    
    # =============================================================================
    # Définir la fonction d'équilibre
    # =============================================================================
    # with st.sidebar:
    #     sidebar = option_menu(None, ['Settings'], 
    #         icons=['sliders'], menu_icon="cast", default_index=0)
    
    # if sidebar == 'Settings':
    #     with st.sidebar.form("my_form"):
    #         #st.write("Définition des paramètres")
    #         M_ben = 78.11
    #         M_tol = 92.14
    #         alpha = st.text_input("Relative volatility (α)", 2.0)
    #         alpha = float(alpha)
    #         q = st.number_input("Fraction of liquid in the feed stream (q)", value= 0.0)
    #         q = float(q)
    #         if q==1:
    #             q=0.9999999999
    #         F = st.text_input("Feed (Kmol/h)", 10.0)
    #         F = float(F)
    #         X_F = st.slider("Mole fraction of the most volatile component in the feed (XF)",min_value=0.0, max_value=100.0, step=0.1, value=40.0)
    #         X_F = X_F/100
    #         X_F_mass = X_F*M_ben/(X_F*M_ben + (1-X_F)*M_tol)
            
    #         X_D = st.slider("Mole fraction of the most volatile component in the distillate (XD)",min_value=0.0, max_value=100.0, step=0.1, value=99.2)
    #         X_D = X_D/100
    #         X_D_mass = X_D*M_ben/(X_D*M_ben + (1-X_D)*M_tol)
            
    #         X_W = st.slider("Mole fraction of the most volatile component in the bottom (XB)",min_value=0.0, max_value=100.0, step=0.1, value=01.4)
    #         X_W = X_W/100
    #         X_W_mass = X_W*M_ben/(X_W*M_ben + (1-X_W)*M_tol)
            
    #         F_mass = X_F*F*M_ben + (1-X_F)*F*M_tol
    
    #         # Every form must have a submit button.
    #         submitted = st.form_submit_button("Run")
    
    def equi(alpha):
        x_eq = np.linspace(0, 1, 101)
        y_eq = alpha*x_eq/(1+(alpha-1)*x_eq)
        return x_eq, y_eq
    
    x_eq, y_eq = equi(alpha)

    st.subheader("1. Minimum Number of Trays at Total Reflux Ratio")
    st.write("- **Shortcut Distillation Method**")
    st.markdown("The Fenske equation determines the minimum number of equilibrium stages, $N_{min}$, For a binary component distillation, the minimum number of theoretical trays can be expressed, such as")
    st.write(r'### <p style="text-align: center;">$$ N_{min} =\frac {ln[(X_D/(1-X_D))((1-X_B)/X_B)]}{log(\alpha)} \hspace*{0.4cm}(1)$$</p>', unsafe_allow_html=True)
    
    N_min = mt.log((X_D/(1-X_D))*((1-X_W)/X_W))/mt.log(alpha)
    STG = N_min
    Stages_min_cal = int(N_min)+1
    
    col_N_min = st.columns(3)
    with col_N_min[1]:
        st.write(r''' $$N_{min} =$$''', round(N_min,3))
        
    # st.write(r''' $$\hspace*{5.4cm} N_{min} =$$''', round(N_min,3))
    st.write("The Minimum Number of Trays is", Stages_min_cal,"stages.")
    
    st.write("- **MacCabe-Thiele Method**")
    
    st.markdown("The equilibrium curve determined by")
    st.write(r'### <p style="text-align: center;">$$ Y_{eq} =\frac {\alpha X_{eq}}{1 + X_{eq}(\alpha - 1)} \hspace*{0.4cm}(2)$$</p>', unsafe_allow_html=True)
    
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
    fig = plt.figure(num=None, figsize=(8, 6))
    for label, x, y in zip(stages, x_s, y_s):
        plt.annotate(label,
                      xy=(x, y),
                      xytext=(0,5),
                      textcoords='offset points', 
                      ha='right')
    
    plt.grid(linestyle='dotted')
    plt.plot(x_eq,y_eq,'-', label="Equilibrium curve")
    plt.plot([0, 1],[0, 1],'black')
    
    plt.scatter(X_D,X_D, color='r', s=20)
    plt.scatter(X_F,X_F, color='r', s=20)
    plt.scatter(X_W,X_W, color='r', s=20)
    
    plt.plot(S[:,0],S[:,1],'r-.', label="Stages")
    
    plt.legend(loc="upper left")
    plt.xlabel("x (-)")
    plt.ylabel("y (-)")
    
    st.pyplot()
    
    Stages_min = s_rows -1
    
    col_N = st.columns(3)
    with col_N[1]:
        st.write(r''' $$N_{min} =$$''', Stages_min)
        
    # st.write(r''' $$\hspace*{5.4cm} N_{min} =$$''', Stages_min)
    st.write("The Minimum Number of Trays is", Stages_min,"stages.")
    
    
    # =============================================================================
    # Rmin
    # =============================================================================
    st.subheader(r'2.  Minimum Reflux Ratio')
    
    st.write("- **Shortcut Distillation Method**")
    st.markdown("An approximate method for calculating the minimum reflux ratio, $R_{min}$, is given by Underwood equations.")
    
    def phi(X):
        return (alpha*X_F)/(alpha-X) + (1-X_F)/(1-X) +q -1
    phi_ = fsolve(phi, 1.66)
    R_min_cal = (alpha*X_D)/(alpha-phi_) + (1-X_D)/(1-phi_) -1
    R_min_cal = R_min_cal[0]
    
    st.write(r'### <p style="text-align: center;">$$ 1 - q = \frac {\alpha X_F}{\alpha - \phi} + \frac {1 - X_F}{1 - \phi} \hspace*{0.4cm}(3)$$</p>', unsafe_allow_html=True)
    st.write(r'### <p style="text-align: center;">$$ R_{min} + 1 = \frac {\alpha X_D}{\alpha - \phi} + \frac {1 - X_D}{1 - \phi} \hspace*{0.4cm}(4)$$</p>', unsafe_allow_html=True)
    
    col_R = st.columns(3)
    with col_R[1]:
        st.write(r''' $$(3)\rightarrow \phi = $$''', round(phi_[0],3))
        st.write(r''' $$(4)\rightarrow R_{min} = $$''', round(R_min_cal,3))
        
    # st.write(r''' $$\hspace*{4.5cm} (3)\rightarrow \phi = $$''', round(phi_[0],3))
    # st.write(r''' $$\hspace*{4.5cm} (4)\rightarrow R_{min} = $$''', round(R_min_cal,3))
    
    More_1 = st.expander("More")
    with More_1:
        st.write("The term $$\phi$$ has the dimension of a constant of relative volatility.") 
    
    
    st.write("- **MacCabe-Thiele Method**")
    
    st.write("The Feed operating line")
    st.write(r'### <p style="text-align: center;">$$ Y = \frac {q}{q - 1} X - \frac {1}{q - 1} X_F \hspace*{0.4cm}(5)$$</p>', unsafe_allow_html=True)
    
    st.write("The rectifying section operating line")
    st.write(r'### <p style="text-align: center;">$$ Y = \frac {R}{R + 1} X + \frac {1}{R + 1} X_D \hspace*{0.4cm}(6)$$</p>', unsafe_allow_html=True)
    
    
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
    fig = plt.figure(num=None, figsize=(8, 6))
    plt.grid(visible=True, which='major',linestyle=':',alpha=0.6)
    plt.grid(visible=True, which='minor',linestyle=':',alpha=0.3)
    plt.minorticks_on()
    #plt.title('Distillation Column Design (MacCabe-Thiele Method)')
    
    plt.plot(x_eq,y_eq,'-', label="Equilibrium curve")
    plt.plot([0, 1],[0, 1],'black')
    
    plt.scatter(X_D, X_D, color='r', s=20)
    plt.text(X_D+0.012,X_D-0.06,'($X_{D},X_{D}$)',horizontalalignment='center')
    plt.scatter(X_F, X_F, color='r', s=20)
    plt.scatter(x_ae, y_ae, color='r', s=20)
    
    plt.plot(x_Rmin, y_Rmin, label="Rectifying section operating line")
    plt.plot(x_fed,y_fed, label="Feed operating line")
    
    plt.legend(loc="best")
    plt.xlabel("x (-)")
    plt.ylabel("y (-)")
    
    plt.scatter(0,ordo, color='r', s=20)
    plt.text(0.02,ordo-0.06,'(0, $\\frac{X_{D}}{R_{min}+1}$)',horizontalalignment='center')
    st.pyplot()
    
    col_RR = st.columns(3)
    with col_RR[1]:
        st.write("$\\frac{X_{D}}{R_{min}+1} =$",round(ordo,3))
        st.write("$R_{min} =$",round(R_min,3))
    
    # st.write("$\hspace*{5.2cm} \\frac{X_{D}}{R_{min}+1} =$",round(ordo,3))
    # st.write("$\hspace*{5.2cm} R_{min} =$",round(R_min,3))
    
    
    st.subheader("3. The actual number of trays")
    col1 = st.columns(2)
    with col1[1]:
        Coeff = st.slider("Select a reflux ratio Coeff",min_value=1.0, max_value=3.0, step=0.01, value=1.106)
        R = Coeff*R_min
    
    with col1[0]:
        st.markdown("The actual reflux ratio, R")
        st.markdown("$R = Coeff \\times R_{min}$")
    st.write("$R =$",round(R,2))
        
    
    st.write("- **Shortcut Distillation Method**")
    st.markdown("Gilliland correlation calculates the number of equilibrium stages (N).")
    st.write(r'### <p style="text-align: center;">$$\frac {N - N_{min}}{N + 1}=0.75 \bigg( 1- \Big(\frac {R - R_{min}}{R + 1}\Big)^{0.566}\bigg) \hspace*{0.4cm}(7)$$</p>', unsafe_allow_html=True)
    
    def equi_sta(N):
        return (N-N_min)/(N+1)-0.75*(1-((R-R_min_cal)/(R+1))**0.566)
    
    N = fsolve(equi_sta, 0)
    N_cal = N[0]
    STG_cal = N_cal
    
    col_NN = st.columns(3)
    with col_NN[1]:
        st.write("$N =$",round(N_cal,3))
    # st.write("$\hspace*{5.2cm} N =$",round(N_cal,3))
    
    N_cal = int(N_cal)+1
    st.write(" Actual number of stages is", N_cal,"stages.")
    
    st.write("- **MacCabe-Thiele Method**")
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
    
    fig = plt.figure(num=None, figsize=(8, 6))
    
    for label, x, y in zip(stages, x_s, y_s):
        plt.annotate(label,
                      xy=(x, y),
                      xytext=(0,5),
                      textcoords='offset points', 
                      ha='right')
    
    plt.grid(linestyle='dotted')
    plt.title('Distillation Column Design (MacCabe-Thiele Method)')
    
    plt.plot(x_eq,y_eq,'-', label="Equilibrium curve")
    plt.plot([0, 1],[0, 1],'black')
    
    
    plt.scatter(X_D,X_D, color='r', s=20)
    plt.scatter(X_F,X_F, color='r', s=20)
    plt.scatter(X_W,X_W, color='r', s=20)
    
    plt.scatter(x_inter,y_inter, color='black', s=20)
    plt.plot(x_alim, y_alim, label="Feed operating line")
    plt.plot(x_appau, y_appau, label="Stripping section operating line")
    plt.plot(x_rect, y_rect, label="Rectifying section operating line")
    # plt.plot(x_fed,y_fed, color='black' )
    
    plt.plot(S[:,0],S[:,1],'-.', label="Stages")
    
    plt.legend(loc="upper left")
    plt.xlabel("x (-)")
    plt.ylabel("y (-)")
    st.pyplot()
    
    N =  s_rows -1
    
    col_NNN = st.columns(3)
    with col_NNN[1]:
        st.write(r''' $$N =$$''', N)
        
    # st.write(r''' $$\hspace*{5.2cm} N =$$''', N)
    st.write(" Actual number of stages is", N,"stages.")
    
    
    
    st.subheader("4. Optimum feed tray location")
    st.markdown("Using Kirkbride method$$~^1~$$:")
    st.write(r'### <p style="text-align: center;">$$\ln(\frac {N_{D}}{N_B})=0.206 \ln \bigg( \Big(\frac {1 - X_{F}}{X_F}\Big) \Big(\frac {X_B}{1 - X_D}\Big)^{2} \Big(\frac {B}{D}\Big) \bigg) \hspace*{0.4cm}(8)$$</p>', unsafe_allow_html=True)
        
    def materiel_balance():
        D = F * (X_F - X_W)/(X_D - X_W)
        B = F - D
        ln_rap = 0.206 * mt.log(((1-X_F)/X_F) * (X_W/(1-X_D))**2 * (B/D))
        rap = mt.exp(ln_rap)
        N_D = N * rap/(rap+1)
        return N_D, rap, D, B
    
    N_F = int(materiel_balance()[0])+1
    STG_F = materiel_balance()[0]
    rap = materiel_balance()[1]
    D = materiel_balance()[2]
    B = materiel_balance()[3]
    
    D_mass = X_D*D*M_LK + (1-X_D)*D*M_HK
    B_mass = X_W*B*M_LK + (1-X_W)*B*M_HK
    
    col_F = st.columns(3)
    with col_F[1]:
        st.write(r''' $$\frac {N_D}{N_B} =$$''', round(rap, 2))
        
    # st.write(r''' $$\hspace*{5.2cm} \frac {N_D}{N_B} =$$''', round(rap, 2))
    
    More_3 = st.expander("More")
    with More_3:
        st.write("$$N_D$$: trays above the feed tray") 
        st.write("$$N_B$$: trays below the feed tray") 
        st.write("$$N_D + N_B = N$$") 
        st.write("$$D$$: Molar flow rate (Distillate)")
        st.write("$$B$$: Molar flow rate (Bottoms)")
        
    st.write(" The number of stages above the feed tray ($N_D$) is",N_F,"trays counted from the top down; this means that the feed tray is the tray number", N_F,".")
    
    
    st.subheader("5. Results")
    
    st.write("- **Distillation Column**")
    
    N_min_cal = str(Stages_min_cal)
    N_min = str(Stages_min)
    
    df = pd.DataFrame(
        np.array([[ str("Reflux Ratio"),  round(R, 2),  round(R, 2)],
                  [ str("Minimum Reflux Ratio"),  round(R_min_cal, 2),  round(R_min, 2)],
                  [str("Minimum Reflux Ratio"),  round(STG,2),  N_min],
                  [str("Number of Trays"), round(STG_cal, 2),  round(N, 2)],
                  [str("Optimal Feed Stage"), round(STG_F, 2),  round(s_f[0], 2)]]),
        columns=('','Shortcut Distillation','MacCabe-Thiele'))
    
    # df = pd.DataFrame(
    #     np.array([[ str("Shortcut Distillation Method"),  N_min_cal,  round(R_min_cal, 4) , round(N_cal, 2), round(N_F, 2)],
    #               [ str("MacCabe-Thiele Method"),  N_min,  round(R_min, 4) , round(N, 2), round(s_f[0], 2)]]),
    #     columns=('','Minimum trays','Minimum Reflux Ratio','Number of Trays','Feed tray'))
    
    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(df)
    
    
    st.write("- **Material Streams**")
    
    df = pd.DataFrame(
        np.array([[ str("Molar Flow"),  str("Kgmole/h") , round(F, 3), round(D, 3), round(B, 3)],
                  [ str("Mass Flow"),  str("Kg/h") , round(F_mass, 3), round(D_mass, 3), round(B_mass, 3)],
                  [str("Mole Frac (LK)"),  str("-") , round(X_F, 3), round(X_D, 3), round(X_W, 3)],
                  [str("Mole Frac (HK)"),  str("-") , round(1-X_F, 3), round(1-X_D, 3), round(1-X_W, 3)],
                  [str("Mass Frac (LK)"),  str("-") , round(X_F_mass, 3), round(X_D_mass, 3), round(X_W_mass, 3)],
                  [str("Mass Frac (HK)"),  str("-") , round(1-X_F_mass, 3), round(1-X_D_mass, 3), round(1-X_W_mass, 3)]]),
        columns=('','Unite','Feed','Distillate','Bottoms'))
    
    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(df)
    
    st.write("$$~^*~$$: Computer Methods in Chemical Engineering Second Edition")
    # st.write("$$~^2~$$: Materiel balnce")
     
        

    




# =============================================================================
#    Simulation
# =============================================================================
elif selected == "Simulation":
    
    HTML_BANNER = """
        <h1 style="color:#FF4B4B;text-align:center;">Distillation Column Simulation</h1>
        <p style="color:#FF4B4B;text-align:center;">Aspen Hysys & DWSIM</p>
        </div>
        """
    stc.html(HTML_BANNER)
    
    st.write(" Using Hysys or DWSIM shortcut distillation method, select Peng-Robinson (PR) as the fluid package, connect feed and product streams, distillate and product streams, condenser and reboiler energy streams, and fully specifying feed stream. While on Design/Parameters page, the following data should be made available:")
    st.write("* LK in the bottom: Benzene with mole fraction 0.014")
    st.write("* HK in distillate: Toluene, mole fraction 0.008")
    st.write("* Condenser and reboiler pressure: 1 atm")
    st.write("* Reflux ratio: 2")
    
    st.subheader("Aspen Hysys Shortcut Distillation Column")
    st.image("HYSYS_Simulation.jpg")
    
    st.subheader("DWSIM Shortcut Distillation Column")
    st.image("DWSIM_Simulation.jpg")





st.markdown('---')
# =============================================================================
# Contact
# =============================================================================
Button = st.expander("Get In Touch With Me!")
with Button:
    col31, col32, col33 = st.columns(3)
    col31.write("[Zakaria NADAH](https://www.linkedin.com/in/zakaria-nadah-00ab81160/)")
    col31.write("Process Engineer")
    col31.write("+336.28.80.13.40")
    col31.write("zakariaenadah@gmail.com")
    
    col33.image("profil_.jpg")