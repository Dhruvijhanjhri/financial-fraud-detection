import streamlit as st


def apply():
    css = """
    <style>
    /* global gradient background */
    body, .reportview-container .main .block-container{
        background: linear-gradient(135deg, #001f3f 0%, #000000 100%) !important;
        color:#ffffff;
    }
    /* glassmorphism card */
    .glass-card{
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding:20px;
        box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);
        transition:transform .3s;
    }
    .glass-card:hover{transform:scale(1.05);}
    .hero{padding:80px 0;text-align:center;}
    h1,h2,h3,h4,h5{font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;}
    .btn{
        background:linear-gradient(90deg,#0af,#06f);
        border:none;color:#fff;padding:12px 30px;border-radius:25px;
        cursor:pointer;font-size:18px;transition:opacity .3s;
    }
    .btn:hover{opacity:.8;}
    .progress-bar{width:80%;margin:0 auto;background:#333;border-radius:10px;}
    .progress-fill{display:block;height:20px;border-radius:10px;background:#0f0;width:0;
        animation: fill 2s forwards;}
    @keyframes fill{from{width:0;} to{width:var(--pct);} }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
