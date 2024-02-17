import os
import pandas as pd
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import optimize, stats
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import t
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
from scipy.optimize import rosen, shgo
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.font_manager as fm

def fnExistFolder(foldername):
    if os.path.exists(foldername) and os.path.isdir(foldername):
        pass  # 폴더가 이미 존재하는 경우 아무 작업도 하지 않습니다.
    else:
        print(f'[{foldername}] folder is created')
        os.makedirs(foldername)

# 주어진 날짜 형식에 따라 연도 추출, 요일 데이터 처리함
def  fn_CPM_date_index(Y):
    # 날짜 형식 확인 및 연도 추출
    if len(Y['USE_DATE'][0]) == 6:
        y = [datetime.strptime(date, "%Y%m").year for date in Y['USE_DATE']]
    elif len(Y['USE_DATE'][0]) == 8:
        y = [datetime.strptime(date, "%Y%m%d").year for date in Y['USE_DATE']]
    else:
        raise ValueError('오류: "Y.USE_DATE"의 크기를 확인하세요.')
    
    date_start = min(set(y))
    date_end = max(set(y))
    yrs_sec = list(range(date_start, date_end + 1))

    if len(Y['USE_DATE'][0]) == 6:
        # 연도별 색상 구분
        date_index = y
    elif len(Y['USE_DATE'][0]) == 8:
        # 요일 데이터
        T_days = pd.DataFrame()
        for year in yrs_vec:
            fnm = f'T_oj_day_holi_{year}.csv'
            T_days0 = pd.read_csv(f'2022 업데이트 깃허브버전\\data input\\weather and holidays\\{fnm}', encoding = 'cp949')
            T_days = pd.concat([T_days, T_days0])

        Day_index = T_days['Day_index']  # 1 평일, 2 토요일, 3 일요일 및 휴일

        idx_week = Day_index == 1
        idx_weekend = Day_index != 1

        date_index = T_days['DayName']

    return date_index

# 주어진 날짜 형식에 따라 연도 추출, 요일 데이터 처리함
def  fn_CPM_date_index(Y):
    # 날짜 형식 확인 및 연도 추출
    if len(Y['USE_DATE'][0]) == 6:
        y = [datetime.strptime(date, "%Y%m").year for date in Y['USE_DATE']]
    elif len(Y['USE_DATE'][0]) == 8:
        y = [datetime.strptime(date, "%Y%m%d").year for date in Y['USE_DATE']]
    else:
        raise ValueError('오류: "Y.USE_DATE"의 크기를 확인하세요.')
    
    date_start = min(set(y))
    date_end = max(set(y))
    yrs_sec = list(range(date_start, date_end + 1))

    if len(Y['USE_DATE'][0]) == 6:
        # 연도별 색상 구분
        date_index = y
    elif len(Y['USE_DATE'][0]) == 8:
        # 요일 데이터
        T_days = pd.DataFrame()
        for year in yrs_vec:
            fnm = f'T_oj_day_holi_{year}.csv'
            T_days0 = pd.read_csv(f'2022 업데이트 깃허브버전\\data input\\weather and holidays\\{fnm}', encoding = 'cp949')
            T_days = pd.concat([T_days, T_days0])

        Day_index = T_days['Day_index']  # 1 평일, 2 토요일, 3 일요일 및 휴일

        idx_week = Day_index == 1
        idx_weekend = Day_index != 1

        date_index = T_days['DayName']

    return date_index

def fn_CPM_1p(x, Tout) :
    # matlab x(1) = x[0] 동일 역할
    b0 = x[0]
    if len(Tout.shape) == 1:
        Tout = Tout.reshape(1, -1)
    nrow = Tout.shape[0] 
    ncol = Tout.shape[1]
    AA = np.ones((nrow, ncol)) #  크기가 (nrow, ncol)인 행렬 AA를 생성하고, 모든 요소를 1로 채움
    Y = b0 * AA
    return Y

def fn_CPM_2p_c(x, Tout):
    b0 = x[0]
    b1 = x[1]
    if len(Tout.shape) == 1:
        Tout = Tout.reshape(1, -1)
    nrow = Tout.shape[0] 
    ncol = Tout.shape[1] # 행렬의 행과 열의 개수를 각각 nrow, ncol에 저장함
    AA = np.ones((nrow, ncol)) # 행렬 AA 생성함
    Y = b0 * AA + b1 * Tout

    return Y

def fn_CPM_2p_h(x, Tout):
    b0 = x[0]
    b1 = x[1]
    if len(Tout.shape) == 1:
        Tout = Tout.reshape(1, -1)
    nrow = Tout.shape[0] 
    ncol = Tout.shape[1] # 행렬의 행과 열의 개수를 각각 nrow, ncol에 저장함
    AA = np.ones((nrow, ncol)) # 행렬 AA 생성함

    Y = b0 * AA - b1 * Tout

    return Y

def fn_CPM_3p_c(x, Tout):
    b0 = x[0]
    b1 = x[1]
    b2 = x[2]
    if len(Tout.shape) == 1:
        Tout = Tout.reshape(1, -1)
    nrow = Tout.shape[0] 
    ncol = Tout.shape[1] 

    # 빈행렬 생성
    Y1 = np.zeros((nrow, ncol)) 
    Y2 = np.zeros((nrow, ncol))

    idx1 = np.where(b2 <= Tout)
    idx2 = np.where(b2 > Tout)

    Y1[idx1] = b0 + b1 * (Tout[idx1] - b2)
    Y2[idx2] = b0

    Y = Y1 + Y2

    return Y

def fn_CPM_3p_h(x, Tout):
    b0 = x[0]
    b1 = x[1]
    b2 = x[2]
    if len(Tout.shape) == 1:
        Tout = Tout.reshape(1, -1)
    nrow = Tout.shape[0] 
    ncol = Tout.shape[1]

    Y1 = np.zeros((nrow, ncol))
    Y2 = np.zeros((nrow, ncol))

    idx1 = np.where(b2 >= Tout)
    idx2 = np.where(b2 < Tout)

    Y1[idx1] = b0 + b1 * (b2 - Tout[idx1])
    Y2[idx2] = b0

    Y = Y1 + Y2

    return Y

def fn_CPM_4p_c(x, Tout):
    b0 = x[0]
    b1 = x[1]
    b2 = x[2]
    if len(Tout.shape) == 1:
        Tout = Tout.reshape(1, -1)

    nrow = Tout.shape[0] 
    ncol = Tout.shape[1]

    Y1 = np.zeros((nrow, ncol))
    Y2 = np.zeros((nrow, ncol))

    idx1 = np.where(b2 >= Tout)
    idx2 = np.where(b2 < Tout)

    Y1[idx1] = b0 + b1 * (b2 - Tout[idx1])
    Y2[idx2] = b0

    Y = Y1 + Y2

    return Y

def fn_CPM_4p_h(x, Tout):
    b0 = x[0]
    b1 = x[1]
    b2 = x[2]
    b3 = x[3]
    if len(Tout.shape) == 1:
        Tout = Tout.reshape(1, -1)

    nrow = Tout.shape[0] 
    ncol = Tout.shape[1]

    Y1 = np.zeros((nrow, ncol))
    Y2 = np.zeros((nrow, ncol))

    idx1 = np.where(b3 >= Tout)
    idx2 = np.where(b3 < Tout)

    Y1[idx1] = b0 - b1 * (Tout[idx1] - b3)
    Y2[idx2] = b0 - b2 * (Tout[idx2] - b3)

    Y = Y1 + Y2

    return Y

def fn_CPM_5p(x, Tout):
    b0 = x[0]
    b1 = x[1]
    b2 = x[2]
    b3 = x[3]
    b4 = x[4]
    if len(Tout.shape) == 1:
        Tout = Tout.reshape(1, -1)

    # Tout
    nrow = Tout.shape[0] 
    ncol = Tout.shape[1]

    Y1 = np.zeros((nrow, ncol))
    Y2 = np.zeros((nrow, ncol))
    Y3 = np.zeros((nrow, ncol))

    idx1 = np.where(b3 >= Tout)
    idx2 = np.where(b4 < Tout)
    idx3 = np.where((b3 < Tout) & (b4 >= Tout))

    Y1[idx1] = b0 + b1 * (b3 - Tout[idx1])
    Y2[idx2] = b0 + b2 * (Tout[idx2] - b4)
    Y3[idx3] = b0 + np.zeros(len(idx3[0]))

    Y = Y1 + Y2 + Y3

    return Y

def fn_CPM_obj(x, Tout, y_mea, CPM_type):
# p변수 0으로 초기화 
    p = 0
# y_pred 빈리스트 생성
    # y_pred = 0
    if CPM_type == '1p':
        p = 1
        y_pred = fn_CPM_1p(x, Tout)

    elif CPM_type == '2p_h':
        p = 2
        y_pred = fn_CPM_2p_h(x, Tout)

    elif CPM_type == '2p_c':
        p = 2
        y_pred = fn_CPM_2p_c(x, Tout)

    elif CPM_type == '3p_h':
        p = 3
        y_pred = fn_CPM_3p_h(x, Tout)

    elif CPM_type == '3p_c':
        p = 3
        y_pred = fn_CPM_3p_c(x, Tout)

    elif CPM_type == '4p_h':
        p = 4

    elif CPM_type == '4p_c':
        p = 4
        y_pred = fn_CPM_4p_c(x, Tout)

    elif CPM_type == '5p':
        p = 5
        y_pred = fn_CPM_5p(x, Tout)

    y_pred = y_pred.reshape(-1)
    n_sample = len(y_pred)
    # 옵션1
    RMSE = np.sqrt(np.sum((y_mea - y_pred) ** 2) / (n_sample - p))
    y_hat = RMSE

    return y_hat

def fn_R2(y_pred, y_mea, p):
    ns = len(y_pred)
    y_avg = np.nanmean(y_mea)
    SSerr = np.nansum((y_pred - y_mea) ** 2)
    SStot = np.nansum((y_mea - y_avg) ** 2)
    
    eee = 0.00001
    R2 = 1 - SSerr / (SStot + eee)
    R2_adj = 1 - (ns - 1) / (ns - p + 1e-10) * (1 - R2**2)
    # R2_adj = 1 - (ns - 1) / (ns - p) * (1 - R2**2)
    
    return R2, R2_adj

def fn_t_score(Tout, y_mea, y_pred, p, yint, yslp):
    x = np.array(Tout)
    x_bar = np.mean(x)
    x_var = np.sum((x - x_bar) ** 2)
    ns = len(x)
    RMSE = np.sqrt(np.sum((y_mea - y_pred) ** 2, axis=0) / (ns - p))
    # Check for variance being zero and handle it to avoid ZeroDivisionError
    if x_var == 0:
        print("Variance of input 'x' is zero, cannot compute standard errors.")
        Se_b0 = np.inf
        Se_b1 = np.inf
    else:
        Se_b0 = RMSE * np.sqrt(1 / ns + (x_bar ** 2 / x_var))
        Se_b1 = RMSE / np.sqrt(x_var)

    # Compute t-values, ensuring that Se_b0 and Se_b1 are not zero or infinity
    if Se_b0 in [0, np.inf]:
        t_c = np.inf
    else:
        t_c = yint / Se_b0

    if Se_b1 in [0, np.inf]:
        t_b = np.inf
    else:
        t_b = yslp / Se_b1

    df = ns - p

    # Calculate p-values, handling infinite t-values appropriately
    if t_c == np.inf:
        pval_c = 0.0
    else:
        pval_c = (1 - t.cdf(np.abs(t_c), df)) * 2

    if t_b == np.inf:
        pval_b = 0.0
    else:
        pval_b = (1 - t.cdf(np.abs(t_b), df)) * 2

    return t_c, t_b, pval_c, pval_b

def fn_cook_d(Tout, y_pred, y_mea, p):
    ns = len(Tout)
    mean_Tout = np.sum(Tout) / ns
    h1 = (Tout - mean_Tout) ** 2
    h2 = np.sum((Tout - mean_Tout) ** 2)
    h3 = 1 / ns
    h = (h1 / h2) + h3
    
    err = (y_pred - y_mea) ** 2
    MSE = np.sum((y_mea - y_pred) ** 2, where=~np.isnan(y_mea)) / (ns - p)
    D = (err) / (MSE * p) * (h / ((1 - h) ** 2))
    
    REF = 3 * np.nanmean(D)
    idx = np.where(D > REF)[0]
    idx = idx + 1
    Cook_out_idx = '/'.join(map(str, idx))
    D = pd.DataFrame([D])
    return Cook_out_idx, D

def fn_ZRE(y_pred, y_mea, p):
    ns = len(y_pred)
    RMSE = np.sqrt(np.sum((y_mea - y_pred) ** 2, where=~np.isnan(y_mea)) / (ns - p))
    Sx = RMSE
    e = (y_pred - y_mea)
    ZRE = (e - np.nanmean(e)) / Sx
    REF = 2.5
    idx = np.where(np.abs(ZRE) > REF)[0]
    idx = idx + 1
    ZRE_out_idx = '/'.join(map(str, idx))

    ZRE = pd.DataFrame([ZRE])
    return ZRE_out_idx, ZRE

def fn_CPM_stat(ID, CPM_type, x_opt, Tout, y_mea, date_start, date_end, T_avg_mon):
    # 데이터 전처리 부분
    idx_y = y_mea == 0
    y_mea[idx_y] = np.nan
# Tout가 nan경우 제외함
    idx = ~np.isnan(Tout) & ~np.isnan(y_mea)
    Tout = Tout[idx]
    y_mea = y_mea[idx]

    # 변수 초기화
    p, p_L, p_R, y_pred, idx_L, idx_R, yint_L, yslp_L, yint_R, yslp_R = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    if CPM_type == '1p': #  Y=b0*AA
        p = p_L = p_R = 1
        y_pred = fn_CPM_1p(x_opt, Tout)
        y_pred = y_pred.reshape(-1)
        idx_L = np.where(Tout <= 50)[0]
        idx_R = np.where(Tout > -50)[0]
        yint_L = x_opt[0]
        yslp_L = 0
        yint_R = 0
        yslp_R = 0
    elif CPM_type == '2p_h': #  Y= b0*AA - b1*Tout
        p = p_L = p_R = 2
        y_pred = fn_CPM_2p_h(x_opt, Tout)
        y_pred = y_pred.reshape(-1)
        idx_L = np.where(Tout <= 50)[0]
        idx_R = np.where(Tout > -50)[0]
        yint_L = x_opt[0]
        yslp_L = -x_opt[1]
        yint_R = 0
        yslp_R = 0
    elif CPM_type == '2p_c': # Y=b0*AA + b1*Tout
        p = p_L = p_R = 2
        y_pred = fn_CPM_2p_c(x_opt, Tout)
        y_pred = y_pred.reshape(-1)
        idx_L = np.where(Tout <= 50)[0]
        idx_R = np.where(Tout > -50)[0]
        yint_L = 0
        yslp_L = 0
        yint_R = x_opt[0]
        yslp_R = x_opt[1]
    elif CPM_type == '3p_h': # Y1(idx1) = b0 + b1*(b2-Tout(idx1))
        # p : 총 파라미터 개수
        # p_L : 왼쪽(난방) 파라미터 개수 
        # p_R : 오른쪽(베이스) 파라미터 개수
        p, p_L, p_R = 3, 2, 1
        y_pred = fn_CPM_3p_h(x_opt, Tout)
        y_pred = y_pred.reshape(-1)
        idx_L = np.where(Tout < x_opt[2])[0]
        idx_R = np.where(Tout >= x_opt[2])[0]
        yint_L = x_opt[0] + x_opt[1] * x_opt[2]
        yslp_L = -x_opt[1]
        yint_R = x_opt[0]
        yslp_R = 0
    elif CPM_type == '3p_c': # Y1(idx1) = b0 + b1*(Tout(idx1)-b2)
        p, p_L, p_R = 3, 1, 2
        y_pred = fn_CPM_3p_c(x_opt, Tout)
        y_pred = y_pred.reshape(-1)
        idx_L = np.where(Tout <= x_opt[2])[0]
        idx_R = np.where(Tout > x_opt[2])[0]
        yint_L = x_opt[0]
        yslp_L = 0
        yint_R = x_opt[0] - x_opt[1] * x_opt[2]
        yslp_R = x_opt[1]
    elif CPM_type == '4p_h':
        # b0_0 : 상수항
        # b1_0 : 기울기 소
        # b2_0 : 기울기 대
        # b3_0 : 변곡점
        # Y1(idx1) = b0 - b1*(Tout(idx1)-b3); 
        # Y2(idx2) = b0 - b2*(Tout(idx2)-b3);
        p, p_L, p_R = 4, 2, 2
        y_pred = fn_CPM_4p_h(x_opt, Tout)
        y_pred = y_pred.reshape(-1)
        idx_L = np.where(Tout <= x_opt[3])[0]
        idx_R = np.where(Tout > x_opt[3])[0]
        yint_L = x_opt[0] + x_opt[1] * x_opt[3]
        yslp_L = -x_opt[1]
        yint_R = x_opt[0] + x_opt[2] * x_opt[3]
        yslp_R = -x_opt[2]
    elif CPM_type == '4p_c':
        p, p_L, p_R = 4, 2, 2
        y_pred = fn_CPM_4p_c(x_opt, Tout)
        y_pred = y_pred.reshape(-1)
        idx_L = np.where(Tout <= x_opt[3])[0]
        idx_R = np.where(Tout > x_opt[3])[0]
        yint_L = x_opt[0] - x_opt[1] * x_opt[2]
        yslp_L = x_opt[1]
        yint_R = x_opt[0] - x_opt[2] * x_opt[3]
        yslp_R = x_opt[2]
    elif CPM_type == '5p':
        p, p_L, p_R = 5, 2, 2
        y_pred = fn_CPM_5p(x_opt, Tout)
        y_pred = y_pred.reshape(-1)
        idx_L = np.where(Tout <= x_opt[3])[0]
        idx_R = np.where(Tout > x_opt[4])[0]
        yint_L = x_opt[0] + x_opt[1] * x_opt[3]
        yslp_L = -x_opt[1]
        yint_R = x_opt[0] - x_opt[2] * x_opt[4]
        yslp_R = x_opt[2]


    ns = len(Tout)
    RMSE = np.sqrt(np.nansum((y_mea - y_pred) ** 2) / (ns - p))

    y_avg = np.nanmean(y_mea)
    NMBE = np.nansum(y_mea - y_pred) / (ns - p) / y_avg * 100

    CVRMSE = RMSE / abs(y_avg) * 100

    # R2 계산 함수 호출
    R2, R2_adj = fn_R2(y_pred, y_mea, p)
   
    idx_A = np.arange(len(Tout))  # This should be 0 to len(Tout)-1
    C = np.setxor1d(idx_A, np.concatenate((idx_L, idx_R)))

# Validate C before using it to index other arrays
    C = C[C < len(Tout)]  # This ensures that all indices in C are within bounds

    try:
        _, _, pval_c_C, _ = fn_t_score(Tout[C], y_mea[C], y_pred[C], 1, x_opt[0], 0)
    except IndexError as e:
        print(f"An index error occurred: {e}")

      # t-score, pval
    _, _, _, pval_b_L = fn_t_score(Tout[idx_L], y_mea[idx_L], y_pred[idx_L], p_L, yint_L, yslp_L)

    # 오른쪽 b4 < Tout
    _, _, _, pval_b_R = fn_t_score(Tout[idx_R], y_mea[idx_R], y_pred[idx_R], p_R, yint_R, yslp_R)

    # 가운데 setxor
    # idx_A = np.arange(1, len(Tout) + 1)
    idx_A = np.arange(len(Tout))  # 0 to len(Tout)-1, which matches the index range of Python arrays
    C = np.setxor1d(idx_A, np.concatenate((idx_L, idx_R)))
    _, _, pval_c_C, _ = fn_t_score(Tout[C], y_mea[C], y_pred[C], 1, x_opt[0], 0)

    # 결과 저장
    ttest_cell = [pval_b_L, pval_b_R, pval_c_C]

    # Cook's 거리 계산 함수 호출
    Cook_out_idx, D = fn_cook_d(Tout, y_pred, y_mea, p)
    # D = pd.DataFrame(D)

    # 표준화 잔차 계산 함수 호출
    ZRE_out_idx, ZRE = fn_ZRE(y_pred, y_mea, p)

    # 저장
    x_opt = x_opt.tolist()  # Convert numpy array to list
    if len(x_opt) < 5:
        for m in range(5 - len(x_opt)):
            x_opt.append(0)

    b0, b1, b2, b3, b4 = x_opt[0:5]

    if CPM_type == '1p' or CPM_type == '2p_h' or CPM_type == '2p_c':
        P_M1 = 0
        P_M2 = 0
    elif CPM_type == '3p_h' : 
        idx = np.argmin(np.abs(Tout - b2))
        P_M1 = idx # 가장 값이 가까운 월
        P_M2 = 0
    elif CPM_type == '3p_c' :
        idx = np.argmin(np.abs(Tout - b2))
        P_M1 = idx
        P_M2 = 0
    elif CPM_type == '4p_h' :
        idx = np.argmin(np.abs(Tout - b2))
        P_M1 = idx
        P_M2 = 0
    elif CPM_type == '4p_c':
        idx = np.argmin(np.abs(Tout - b2))
        P_M1 = idx
        P_M2 = 0
    elif CPM_type == '5p':
        idx1 = np.argmin(np.abs(Tout - b3))
        idx2 = np.argmin(np.abs(Tout - b4))
        P_M1 = idx1
        P_M2 = idx2
    ####### 냉난방기저 비율 및 월별 에너지량 계산 ######
    ## b0 ~ b4에 더해서 냉난방 기저비율 도출해서 사용해보려 하였음
    # 면적 == 에너지 사용량이라 생각했음
    T_avg = np.zeros(12)

    for i in range(12):
        T_avg[i] = T_avg_mon[i]

    EUI_mon = np.zeros(12)  # Monthly energy usage

    if CPM_type == '1p':
        T0_max = np.max(Tout)
        T0_min = np.min(Tout)

        ratio_c = 0
        ratio_h = 0
        ratio_b = 0

        # 최고온도-최저온도 차이가 건물의 기저부하 에너지 사용량에 어떤 영향을 미치는지 평가하는데 활용
        AREA_CPM = b0 * (T0_max - T0_min)

        for i in range(12):
            EUI_mon[i] = b0

    elif CPM_type == '2p_h':
        T0_max = np.max(Tout)
        T0_min = np.min(Tout)

        a = b1 * T0_min + b0
        yi = T0_max * (-b1) + a

        cooling = 0

        if yi > 0:
            heating = (b0 - yi) * (T0_max - T0_min) / 2
            base = yi * (T0_max - T0_min)
        else:
            base = 0
            x0 = -b1 / a
            heating = (x0 - T0_min) * b0 / 2

    # 전체 에너지 사용에서 차지하는 비율을 나타냄
        ratio_c = cooling / (cooling + heating + base) * 100
        ratio_h = heating / (cooling + heating + base) * 100
        ratio_b = base / (cooling + heating + base) * 100

    # 전체 에너지 사용량 계산함
        AREA_CPM = cooling + heating + base

        for i in range(12):
            EUI_mon[i] = -b1 * (T_avg_mon[i] - T0_min) + b0

    elif CPM_type == '2p_c' : 
        T0_max = max(Tout)
        T0_min = min(Tout)

        a = -b1 * T0_min + b0
        yj = T0_max * b1 + a

        cooling = (yj - b0) * (T0_max - T0_min) / 2
        heating = 0
        base = b0 * (T0_max - T0_min)

        ratio_c = (cooling / (cooling + heating + base)) * 100
        ratio_h = (heating / (cooling + heating + base)) * 100
        ratio_b = (base / (cooling + heating + base)) * 100

        AREA_CPM = cooling + heating +base

        for i in range(12):
            EUI_mon[i] = -b1 * (T_avg_mon[i] - T0_min) + b0

    elif CPM_type == '3p_h' :
        # 관찰되는 기간 동안의 최소 온도 설정함
        T0_max = max(Tout)
        T0_min = min(Tout)

        # 총 에너지 사용량 계산
        a = b1 * b2 + b0 # 기저부하 에너지와 관련된 다른 파라미터 조합을 나타냄
        yi = -T0_min * b1 + a # 최소 외부온도에서 예상되는 총 에너지 사용량

        cooling = 0
        heating = (b2 - T0_min) * (yi - b0) / 2 # 난방에 필요한 에너지 사용량 # (b2 - T0_min) 차이가 클수록 난방 에너지 사용량 증가함
        base = b0 * (T0_max - T0_min) # 기저부하에 에너지 사용량 계산

        ratio_c = (cooling / (cooling + heating + base)) * 100
        ratio_h = (heating / (cooling + heating + base)) * 100
        ratio_b = (base / (cooling + heating + base)) * 100

        AREA_CPM = cooling + heating + base # 총 에너지 사용량 계산

        EUI_mon = [] #월별 에너지 사용 강도 계산 
        for i in range(12):
            if T_avg_mon[i] < b2: #월별 온도가 난방 ㅇ
                EUI_mon.append(-b1 * (T_avg_mon[i] - b2) + b0)
            else:
                EUI_mon.append(b0)

    elif CPM_type == '3p_c' : 
        T0_max = max(Tout)
        T0_min = min(Tout)

        a = -b1 * b2 + b0
        yj = T0_max * b1 + a # 최대 외부 온도에서 예상되는 총 에너지 사용량 

        cooling = (yj - b0) * (T0_max - b2) / 2 # 냉방에 필요한 에너지 사용량 계산 # (T0_max - b2) : 이 차이에 따라 냉방 에너지 사용량 증가
        heating = 0
        base = b0 * (T0_max - T0_min)

        ratio_c = (cooling / (cooling + heating + base)) * 100
        ratio_h = (heating / (cooling + heating + base)) * 100
        ratio_b = (base / (cooling + heating + base)) * 100

        AREA_CPM = cooling + heating + base

        EUI_mon = []
        for i in range(12):
            if T_avg_mon[i] > b2:
                EUI_mon.append(b1 * (T_avg_mon[i] - b2) + b0)
            else:
                EUI_mon.append(b0)
        
    elif CPM_type == '4p_h' : 
        ratio_c = 0
        ratio_h = 0
        ratio_b = 0
        AREA_CPM = 0

    elif CPM_type == '4p_c' : 
        ratio_c = 0
        ratio_h = 0
        ratio_b = 0
        AREA_CPM = 0

    elif CPM_type == '5p' :
        T0_max = max(Tout)
        T0_min = min(Tout)

        a1 = b1 * b3 + b0
        yi = -T0_min * b1 + a1

        a2 = -b2 * b4 +b0
        yj = T0_max * b2 + a2

        cooling = (yj - b0) * (T0_max - b4) / 2
        heating = (yi - b0) * (b3 - T0_min) / 2
        base = b0 * (T0_max - T0_min)

        ratio_c = (cooling / (cooling + heating + base)) * 100
        ratio_h = (heating / (cooling + heating + base)) * 100
        ratio_b = (base / (cooling + heating + base)) * 100
        
        AREA_CPM = cooling + heating + base
        
        EUI_mon = []
        for i in range(12):  # 12개월에 대해 반복
            if T_avg_mon[i] < b3:
                EUI_mon.append(-b1 * (T_avg[i] - b3) + b0)
            elif T_avg_mon[i] > b4:
                EUI_mon.append(b2 * (T_avg[i] - b4) + b0)
            else:
                EUI_mon.append(b0)

    EUI_sum = np.sum(EUI_mon)  # Annual energy usage
    print(f"EUI_SUM is : {EUI_mon}")

    data = [ID, CPM_type, b0, b1, b2, b3, b4, RMSE, NMBE, CVRMSE, R2_adj, ratio_c, ratio_h, ratio_b, AREA_CPM,
            EUI_mon[0], EUI_mon[1], EUI_mon[2], EUI_mon[3], EUI_mon[4], EUI_mon[5],
            EUI_mon[6], EUI_mon[7], EUI_mon[8], EUI_mon[9], EUI_mon[10], EUI_mon[11], EUI_sum]
    colnms = ['ID', 'CPM_TY', 'b0','b1','b2','b3','b4', 'RMSE','NMBE', 'CVRMSE',
                'R2', 'ratio_c', 'ratio_h', 'ratio_b', 'AREA_CPM',
                'EUI_mon(1)', 'EUI_mon(2)', 'EUI_mon(3)', 'EUI_mon(4)', 'EUI_mon(5)', 'EUI_mon(6)',
                'EUI_mon(7)', 'EUI_mon(8)', 'EUI_mon(9)', 'EUI_mon(10)', 'EUI_mon(11)', 'EUI_mon(12)', 'EUI_sum']
    T_stat = pd.DataFrame([data], columns=colnms)

    return T_stat, Cook_out_idx, D, ZRE_out_idx, ZRE, p

def fn_CPM_run(Tout, y_mea, pathnm_csv, pathnm_pics, Case_str, date_fr, date_to,T_avg_mon):
    # 문서 정보
    # 작성자: 김덕우
    # 작성일: 191206

    # 주의사항
    # Tout과 y_mea는 1x12n 배열이어야 함.

    # 수정사항
    # (220608) 기울기 제약을 해제. 이상치 탐색시 활용.
    # (210302) 방학등 1% 이하의 사용량은 nan 으로 처리한다.
    # (210302) 4parameter 모델은 고려하지 않음
    # (230522) 입력값 추가: area, y축 EUI 기준으로 변경 (이나윤) -> 28-29줄 생성


    # 폴더 체크
    fnExistFolder(pathnm_pics)

    ## CPM type 선택
    CPM_type_list = ['1p', '2p_h', '2p_c', '3p_h', '3p_c', '5p']

    # 3차 셋팅 (221212),  외기 온도 탐색 범위 제약
    To_lb = 0
    To_ub = 25

    # 회귀 파라메터 제약
    Y_min = np.min(y_mea)
    Y_max = np.max(y_mea)
    
    # b0_0 = median(y_mea, 'omitnan'); y_mea에서 omitnan 옵션사용 => median 계산 
    # b1_0 = (Y_max-Y_min)/100; % % 데이터 정규화, 분할
    # b2_0 = b1_0;


    # 2차셋팅 # 최적화 알고리즘 시작점
    b0_0 = 0
    b1_0 = 0
    b2_0 = 0

    # save_T_stat = np.array([])
    save_T_stat = []
    C_cookd = []
    C_ZRE = []
    # *x : 가변개수의 매개변수를 나타냄 ➡️ 매개변수를 튜플 형태로 함수 내부에서 사용할 수 있음

    for m in range(len(CPM_type_list)):
        CPM_type = CPM_type_list[m]
        
        ## 초기값 제약조건 설정
        # A = [1, 2] 및 b = 1을 사용하여 A*x <= b 형식으로 선형 부등식 제약 조건을 나타냅니다.
        # Aeq = [2, 1] 및 beq = 1을 사용하여 Aeq*x = beq 형식으로 선형 등식 제약 조건을 나타냅니다

        if CPM_type == '1p':
            x0 = np.array([b0_0])
            A = np.array([])
            b = np.array([])
            Aeq = np.array([])
            beq = np.array([])
            lb = [Y_min]
            ub = [Y_max]
            nonlcon = None

        elif CPM_type == '2p_h':
            # b0_0: 좌측 기울기
            # b1_0: 우측 상수항
            x0 = np.array([b0_0, b1_0])
            A = np.array([])
            b = np.array([])
            Aeq = np.array([])
            beq = np.array([])
            lb = [0, Y_min]
            ub = [np.inf, Y_max]
            nonlcon = None

        elif CPM_type == '2p_c':
            # b0_0: 좌측 상수항
            # b1_0: 우측 기울기
            x0 = [b0_0, b1_0]
            A = np.array([])
            b = np.array([])
            Aeq = np.array([])
            beq = np.array([])
            lb = [Y_min, 0]
            ub = [Y_max, np.inf]
            nonlcon = None

        elif CPM_type == '3p_h':
            # b0_0: 상수항
            # b1_0: 기울기
            # b2_0: 변곡점
            x0 = [b0_0, b1_0, (To_lb + To_ub) / 2] #변곡점 온도범위 중간점으로 초기화
            A = np.array([])
            b = np.array([])
            Aeq = np.array([])
            beq = np.array([])
            lb = [Y_min, 0, To_lb]
            ub = [Y_max, np.inf, To_ub]
            nonlcon = None

        elif CPM_type == '3p_c':
            # b0_0: 상수항
            # b1_0: 기울기
            # b2_0: 변곡점
            x0 = [b0_0, b1_0, (To_lb + To_ub) / 2]
            A = np.array([])
            b = np.array([])
            Aeq = np.array([])
            beq = np.array([])
            lb = [Y_min, 0, To_lb]
            ub = [Y_max, np.inf, To_ub]
            nonlcon = None

        elif CPM_type == '5p':
            # b0_0: 상수항
            # b1_0: 좌측 기울기
            # b2_0: 우측 기울기
            # b3_0: 좌측 변곡점
            # b4_0: 우측 변곡점
            x0 = [b0_0, b1_0, b2_0, (To_lb + To_ub) / 2, (To_lb + To_ub) / 2]
            A = np.array([0, 0, 0, 1, -1])
            b = np.array([0])
            Aeq = np.array([])
            beq = np.array([])
            lb = [Y_min, 0, 0, To_lb, To_lb]
            ub = [Y_max, np.inf, np.inf, To_ub, To_ub]
            nonlcon = None
        

        # 최적화

        # 이게 진짜 사용해야할 코드인데 시간이 너무 오래 걸리는 관계로 잠시만 교체~~~~#
        minimizer_kwards = {
                    "method": "trust-constr",
                    "bounds": [(lower_bound, upper_bound) for lower_bound, upper_bound in zip(lb, ub)],
                    # "args": (A, b, Aeq, beq)
                    "args": (Tout, y_mea, CPM_type)}
        # result = minimize(fn_CPM_obj, x0, **minimizer_kwards, options = {'disp' : False} )
        result = basinhopping(fn_CPM_obj,x0, minimizer_kwargs = minimizer_kwards, disp = False )

        # minimizer_kwards = {
        #             "method": "trust-constr",
        #             "bounds": [(lower_bound, upper_bound) for lower_bound, upper_bound in zip(lb, ub)],
        #             # "args": (A, b, Aeq, beq)
        #             "args": (Tout, y_mea, CPM_type)}
        # result = minimize(fn_CPM_obj, x0, **minimizer_kwards, options = {'disp' : False} )

        x_opt = result.x
        fval = result.fun
        # exitflag = result.status
        output = result.message
        
        # 통계추출
        T_stat, Cook_out_idx, D, _, ZRE, p = fn_CPM_stat(Case_str, CPM_type, x_opt, Tout, y_mea, date_fr, date_to,T_avg_mon)

        C_cookd.append(D)
        C_ZRE.append(ZRE)
        # print(f"C_ZRE is {C_ZRE}")
        save_T_stat.append(T_stat)
        
    # 최종 모델 선택
    C_cookd = np.squeeze(C_cookd)
    C_cookd = pd.DataFrame(C_cookd)

    C_ZRE = np.squeeze(C_ZRE)
    C_ZRE = pd.DataFrame(C_ZRE)

    # 배열축소 
    save_T_stat = np.squeeze(save_T_stat)
    save_T_stat = pd.DataFrame(save_T_stat)
    # save_T_stat.columns = ['ID', 'DATE_S', 'DATE_E', 'CPM_TY', 'b0', 'b1', 'b2', 'b3', 'b4', 'ns', 'RMSE', 'NMBE', 'CVRMSE', 'R2_adj',
    #         'pval_b_L', 'pval_b_R', 'pval_c_C', 'Ckd_out_idx', 'ZRE_out_idx', 'P_M1', 'P_M2', 'R2']
    save_T_stat.columns = ['ID', 'CPM_TY', 'b0','b1','b2','b3','b4', 'RMSE','NMBE', 'CVRMSE',
                'R2', 'ratio_c', 'ratio_h', 'ratio_b', 'AREA_CPM',
                'EUI_mon(1)', 'EUI_mon(2)', 'EUI_mon(3)', 'EUI_mon(4)', 'EUI_mon(5)', 'EUI_mon(6)',
                'EUI_mon(7)', 'EUI_mon(8)', 'EUI_mon(9)', 'EUI_mon(10)', 'EUI_mon(11)', 'EUI_mon(12)', 'EUI_sum']

    p1 = np.argsort(save_T_stat['RMSE'].values)
    r1 = np.arange(1, len(save_T_stat) + 1)
    rank_md = r1[p1]
    
    p2 = np.argsort(rank_md)
    r2 = np.arange(1, len(rank_md) + 1)
    MD_RANK = r2[p2]

    # 랭크 저장
    save_T_stat['MD_RANK'] = MD_RANK
    save_T_stat = save_T_stat[['MD_RANK'] + [col for col in save_T_stat.columns if col != 'MD_RANK']]

    return save_T_stat, C_cookd, C_ZRE

import matplotlib.pyplot as plt

def fn_nan2zero(X_array):
    # Convert the input to a numpy array if it's not already
    X_array = np.array(X_array)

    # Find the indices where values are NaN
    idx = np.isnan(X_array)

    # Replace NaN values with zero
    X_array[idx] = 0

    return X_array, idx

def fn_CPM_plot_bestfit(do_plot, save_T_stat, C_cookd, C_ZRE, Tout, y_mea, pathnm_csv, pathnm_pics, Case_str, date_start, date_end, CPM_type_best, date_index):
    # Replace NaN values in y_mea with zeros
    _, idx_nan = fn_nan2zero(y_mea)
    fnExistFolder(pathnm_pics)
    CPM_type_best = ['1p', '2p_h', '2p_c', '3p_h', '3p_c', '5p']

    font_path = 'c:/USERS/ZXORO/APPDATA/LOCAL/MICROSOFT/WINDOWS/FONTS/NANUMGOTHIC.OTF'
    # font_path = 'c:/USERS/ZXORO/APPDATA\LOCAL\MICROSOFT/WINDOWS/FONTS/OWNGLYPH_2022_UWY_SI_WOO-RG.TTF'
    fontprop = fm.FontProperties(fname=font_path, size=12)

    if do_plot == 0:
        print('No CPM plot')
    elif do_plot == 1:
        for CPM_type in CPM_type_best:
            T_stat = save_T_stat[save_T_stat['CPM_TY'] == CPM_type]
            if T_stat.empty:
                print(f"No data found for CPM_type: {CPM_type}")
            else:
                display(T_stat)
            x_opt = [T_stat['b0'].iloc[0], T_stat['b1'].iloc[0], T_stat['b2'].iloc[0], T_stat['b3'].iloc[0], T_stat['b4'].iloc[0]]
            plt.figure(figsize=(5, 2))

            unique_dates = list(set(date_index))  # Replace with your actual data
            cmap = plt.cm.viridis  # You can choose any available colormap

            # Normalize color map based on the range of years
            norm = mcolors.Normalize(vmin=min(unique_dates), vmax=max(unique_dates))

            # Create a scatter plot
            plt.scatter(Tout, y_mea, c=date_index, cmap=cmap, norm=norm)

            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=str(date), markersize=10, 
                                           markerfacecolor=cmap(norm(date))) for date in unique_dates]
            plt.legend(handles=legend_elements, title='Year', loc='center left', bbox_to_anchor=(1, 0.5))


            plt.grid(True)



            # Labels and Title
            plt.xlabel('Temperature [°C]',  fontproperties=fontprop )
            plt.ylabel('EUI [kWh/m^2]',  fontproperties=fontprop )  #  EUI 단위로 표시
            plt.title(f'PK: {Case_str} / TYPE: {CPM_type}\nCVRMSE: {round(float(T_stat.CVRMSE), 1)}',  fontproperties=fontprop )

            if CPM_type == '1p' :
                y_pred = fn_CPM_1p(x_opt, Tout)
                b0 = x_opt[0]

                Xset = np.array(Tout)
                Yset = np.array(y_pred)
                
                Xset_flattened = Xset.flatten() if Xset.ndim > 1 else Xset
                Yset_flattened = Yset.flatten() if Yset.ndim > 1 else Yset

                # Plotting the model prediction
                plt.plot(Xset_flattened, Yset_flattened, 'g.-', linewidth=1.5)
                plt.text(min(Xset_flattened), b0,'b0', horizontalalignment='left', verticalalignment='bottom')

                # Adjusting y-axis limits if necessary
                ax = plt.gca()
                y_ax = ax.get_ylim()
                if y_ax[1] < 1:
                    ax.set_ylim(y_ax[0], 1)
                # plt.show()

            elif CPM_type == '2p_h' : 
                y_pred = fn_CPM_2p_h(x_opt, Tout)
                b0 = x_opt[0]
                b1 = x_opt[1]
                #scatter(Tout, y_mea, 'k')

                Xset = np.array(Tout)
                Yset = np.array(y_pred)
                Xset = Xset.flatten()
                Yset = Yset.flatten()

                I = np.argsort(Xset)
                B = [Xset[i] for i in I]

                # 구역별 플롯 
                # Plotting the model prediction
                plt.plot(B, Yset[I], 'r.-', linewidth=1.5) 
                plt.text(np.median(Xset), np.median(Yset),'b1', horizontalalignment='left', verticalalignment='bottom')
                # plt.text(np.median(Xset), np.median(Yset), 'b1', horizontalalignment='left', verticalalignment='bottom')
                
                # Plotting b0 line
                plt.plot([0], [b0], 'k:.') # b0 라인
                plt.text(0,b0,  'b0', horizontalalignment='left', verticalalignment='bottom')
                # plt.text(0, b0, 'b0', horizontalalignment='right', verticalalignment='bottom')
                
                # Adust y-axis if necessary
                ax = plt.gca()
                y_ax = ax.get_ylim()
                if y_ax[1] < 1:
                    y_ax[1] = 1
                    ax.set_ylim(y_ax)
                # plt.show()

            elif CPM_type == '2p_c' :
                y_pred = fn_CPM_2p_h(x_opt, Tout)
                b0 = x_opt[0]
                b1 = x_opt[1]
                #scatter(Tout, y_mea, 'k')

                Xset = np.array(Tout)
                Xset = Xset.flatten()
                Yset = np.array(y_pred)
                Yset = Yset.flatten()

                I = np.argsort(Xset)
                B = [Xset[i] for i in I]

                # Plotting the model predictioin 
                plt.plot(Tout, y_pred, 'b.-', linewidth=1.5)
                plt.text(np.median(Xset), np.median(Yset), 'b1', horizontalalignment='left', verticalalignment='bottom')
             # plt.text(np.median(Xset), np.median(Yset), 'b1', horizontalalignment='left', verticalalignment='bottom')

                #plotting b0 line 
                plt.plot([0], [b0], 'k:.') # b0라인
                # plt.text(np.min(Xset), b0,  f'b0: {b0:.2f}', horizontalalignment='left', verticalalignment='bottom')
                plt.text(0, b0, 'b0', horizontalalignment='right', verticalalignment='bottom')

                # False line (seems redundant but included for completeness) 
                plt.plot([np.min(Xset), np.min(Xset)], [0, 0], 'k:.') # FALSE LINE

                 # Adust y-axis if necessary
                ax = plt.gca()
                y_ax = ax.get_ylim()
                if y_ax[1] < 1:
                    y_ax[1] = 1
                    ax.set_ylim(y_ax)
                # plt.show()

            elif CPM_type == '3p_h' :
                y_pred = fn_CPM_3p_h(x_opt, Tout)
                b0 = x_opt[0]
                b1 = x_opt[1]
                b2 = x_opt[2]

                Xset = np.append(Tout, b2)
                Yset = np.append(y_pred, b0)

                I = np.argsort(Xset)
                B = [Xset[i] for i in I]
                B = np.array(B)
                BY = Yset[I]
                BY = np.array(BY)

                # Plotting the model predictioin 
                plt.plot(B[B <= b2], BY[B <= b2], 'r.-', linewidth=1.5)
                plt.plot(B[B >= b2], BY[B >= b2], 'g.-', linewidth=1.5)
                plt.text(np.median(B[B < b2]), np.median(BY[BY > b0]), 'b1', horizontalalignment='left', verticalalignment='bottom')
                # plt.text(np.median(B[B < b2]), np.median(BY[BY > b0]), f'b1: {b1:.2f}', horizontalalignment='left', verticalalignment='bottom')

                # Plotting the b2 line
                plt.plot([b2, b2], [0, b0], 'k:.')
                plt.text(b2, b0/2, 'b2', horizontalalignment='left', verticalalignment='bottom')
                # plt.text(b2, b0/2,  f'b2: {b2:.2f}', horizontalalignment='left', verticalalignment='bottom')

                plt.plot([np.min(Xset), np.max(Xset)], [b0, b0], 'k:', linestyle='dotted')
                plt.text(np.min(Xset), b0, 'b0', horizontalalignment='right', verticalalignment='bottom')
                # plt.text(np.min(Xset), b0,  f'b0: {b0:.2f}', horizontalalignment='left', verticalalignment='bottom')

                # Finding the closest month to b2
                idx1 = np.argmin(np.abs(Tout - b2))
                Tout1 = Tout.flatten()
                NR_M1 = Tout1[idx1] # 가장 값이 가까운 월

                plt.plot(NR_M1, 0, 'k:v') # Marker for the closest month 
                plt.text(NR_M1, 0, f'M{idx1 + 1}', horizontalalignment='right', verticalalignment='bottom')

                 # Adust y-axis if necessary
                ax = plt.gca()
                y_ax = ax.get_ylim()
                if y_ax[1] < 1:
                    y_ax[1] = 1
                    ax.set_ylim(y_ax)
                # plt.show()
                    
            elif CPM_type == '3p_c' :
                y_pred = fn_CPM_3p_c(x_opt, Tout)
                b0 = x_opt[0]
                b1 = x_opt[1]
                b2 = x_opt[2]

                Xset = np.append(Tout, b2)
                Yset = np.append(y_pred, b0)
                I = np.argsort(Xset)
                B = [Xset[i] for i in I]
                B = np.array(B)
                BY = Yset[I]
                BY = np.array(BY)

                # Plotting the model predictioin 
                plt.plot(B[B <= b2], BY[B <= b2], 'r.-', linewidth=1.5)
                plt.plot(B[B >= b2], BY[B >= b2], 'g.-', linewidth=1.5)
                plt.text(np.median(B[B > b2]), np.median(BY[BY > b0]), 'b1', horizontalalignment='left', verticalalignment='bottom')
                # plt.text(np.median(B[B < b2]), np.median(BY[BY > b0]), f'b1: {b1:.2f}', horizontalalignment='left', verticalalignment='bottom')

                # Plotting the b2 line
                plt.plot([b2, b2], [0, b0], 'k:.')
                plt.text(b2, b0/2, 'b2', horizontalalignment='left', verticalalignment='bottom')
                # plt.text(b2, b0/2,  f'b2: {b2:.2f}', horizontalalignment='left', verticalalignment='bottom')

                plt.plot([np.min(Xset), np.max(Xset)], [b0, b0], 'k:', linestyle='dotted')
                plt.text(np.min(Xset), b0, 'b0', horizontalalignment='right', verticalalignment='bottom')
                # plt.text(np.min(Xset), b0,  f'b0: {b0:.2f}', horizontalalignment='left', verticalalignment='bottom')

                # Finding the closest month to b2
                idx1 = np.argmin(np.abs(Tout - b2))
                Tout1 = Tout.flatten()
                NR_M1 = Tout1[idx1] # 가장 값이 가까운 월

                plt.plot(NR_M1, 0, 'k:v') # Marker for the closest month 
                plt.text(NR_M1, 0, f'M{idx1+1}', horizontalalignment='right', verticalalignment='bottom')

                 # Adust y-axis if necessary
                ax = plt.gca()
                y_ax = ax.get_ylim()
                if y_ax[1] < 1:
                    y_ax[1] = 1
                    ax.set_ylim(y_ax)
                # plt.show()

            elif CPM_type == '4p_h' :
                y_pred = fn_CPM_4p_h(x_opt, Tout)
                b0 = x_opt[0]
                b1 = x_opt[1]
                b2 = x_opt[2]
                b3 = x_opt[3]

                Xset = np.append(Tout, b3)
                Yset = np.append(y_pred, b0)
                I = np.argsort(Xset)
                B = [Xset[i] for i in I]
                B = np.array(B)
                BY = Yset[I]
                BY = np.array(BY)

                # Conditional Plotting
                plt.plot(B[B <= b3], BY[B <= b3], '.-', color=[0, 0.7, 1], linewidth=1.5)
                plt.text(np.median(B[B <= b3]), np.median(BY[B <= b3]), 'b1', horizontalalignment='left', verticalalignment='top')

                plt.plot(B[B >= b3], BY[B >= b3], 'b.-', linewidth=1.5)
                plt.text(np.median(B[B >= b3]), np.median(BY[B >= b3]), 'b2', horizontalalignment='left', verticalalignment='top')

                # Auxiliary Lines
                plt.plot([np.min(Xset), np.max(Xset)], [b0, b0], 'k:', linestyle='dotted') #b0 라인
                plt.text(np.min(Xset), b0, 'b0', horizontalalignment='right', verticalalignment='bottom')

                plt.plot([b3, b3], [0, b0], 'k:', linestyle='dotted')
                plt.text(b3, b0/2, 'b3', horizontalalignment='right', verticalalignment='bottom')

                # Closest Month to b3
                idx1 = np.argmin(np.abs(Tout - b3))
                Tout1 = Tout.flatten()
                NR_M1 = Tout1[idx1] # 가장 값이 가까운 월
                plt.plot(NR_M1, 0, 'k:v')
     
                plt.text(NR_M1, 0, f'M{idx1+1}', horizontalalignment='right', verticalalignment='bottom')

                 # Adust y-axis if necessary
                ax = plt.gca()
                y_ax = ax.get_ylim()
                if y_ax[1] < 1:
                    y_ax[1] = 1
                    ax.set_ylim(y_ax)
                # plt.show()
            
            elif CPM_type == '5p' :
                y_pred = fn_CPM_5p(x_opt, Tout)
                b0 = x_opt[0]
                b1 = x_opt[1]
                b2 = x_opt[2]
                b3 = x_opt[3]
                b4 = x_opt[4]

                Xset = np.append(Tout, [b3, b4])
                Yset = np.append(y_pred, [b0, b0])
                I = np.argsort(Xset)
                B = [Xset[i] for i in I]
                B = np.array(B)
                BY = Yset[I]
                BY = np.array(BY)

                # Conditional Plotting
                plt.plot(B[B <= b3], BY[B <= b3], 'r.-', linewidth=1.5)
                plt.text(np.median(B[B <= b3]), np.median(BY[B <= b3]), 'b1', horizontalalignment='left', verticalalignment='bottom')
                plt.plot(B[(B >= b3) & (B <= b4)], BY[(B >= b3) & (B <= b4)], 'g.-', linewidth=1.5)
                plt.plot(B[B >= b4], BY[B >= b4], 'b.-', linewidth=1.5)
                plt.text(np.median(B[B >= b4]), np.median(BY[B >= b4]),  'b2', horizontalalignment='left', verticalalignment='top')

                # Auxiliary Lines
                plt.plot([np.min(Xset), np.max(Xset)], [b0, b0], 'k:', linestyle='dotted') #b0 라인
                plt.text(np.min(Xset), b0, 'b0', horizontalalignment='right', verticalalignment='bottom')
                plt.plot([b3, b3], [0, b0], 'k:', linestyle='dotted')
                plt.text(b3, b0/2, 'b3', horizontalalignment='left', verticalalignment='bottom')
                plt.plot([b4, b4], [0, b0], 'k:', linestyle='dotted')
                plt.text(b4, b0/2,  'b4', horizontalalignment='right', verticalalignment='bottom')

                # Closest Months to b3 and b4
                idx1 = np.argmin(np.abs(Tout - b3))
                idx2 = np.argmin(np.abs(Tout - b4))
                Tout1 = Tout.flatten()
                NR_M1 = Tout1[idx1]
                NR_M2 = Tout1[idx2]
                plt.plot(NR_M1, 0, 'k:v')
                plt.text(NR_M1, 0, f'M{idx1+1}', horizontalalignment='center', verticalalignment='top')
                plt.plot(NR_M2, 0, 'k:v')
                plt.text(NR_M2, 0, f'M{idx2+1}', horizontalalignment='center', verticalalignment='top')

                 # Adust y-axis if necessary
                ax = plt.gca()
                y_ax = ax.get_ylim()
                if y_ax[1] < 1:
                    y_ax[1] = 1
                    ax.set_ylim(y_ax)
                # plt.show()
    xticks = ax.get_xticks()
    xticks_range = np.array(list(map(int, np.arange(int(min(xticks)), int(max(xticks)) + 1, 5))))
    ax.set_xticks(xticks_range)




