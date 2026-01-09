import streamlit as st
import time
from nexus_db import get_supabase_client


# --- AUTHENTICATION FLOWS ---

def sign_up_with_email(email, password):
    """Registers a user. Supabase automatically sends a confirmation email."""
    supabase = get_supabase_client()
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"plan_type": "free"}}
        })
        # Check if confirmation email was sent
        if response.user and response.user.identities and len(response.user.identities) > 0:
            return True, "✅ Account created! Please check your email to confirm your account."
        elif response.user:
            return True, "✅ Account created!"
        else:
            return False, "❌ Signup failed."
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def sign_in_with_password(email, password):
    """Standard login."""
    supabase = get_supabase_client()
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            return True, response.user
        return False, None
    except Exception as e:
        return False, str(e)


def sign_in_with_otp(email):
    """Requests a One-Time Password (OTP) via email."""
    supabase = get_supabase_client()
    try:
        supabase.auth.sign_in_with_otp({"email": email})
        return True, "✅ OTP sent to your email."
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def verify_otp(email, token):
    """Verifies the OTP token."""
    supabase = get_supabase_client()
    try:
        response = supabase.auth.verify_otp({"email": email, "token": token, "type": "email"})
        if response.user:
            return True, response.user
        return False, None
    except Exception as e:
        return False, str(e)


def logout():
    supabase = get_supabase_client()
    supabase.auth.sign_out()
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None
    st.rerun()


# --- UI COMPONENTS ---

def login_form():
    st.markdown("## 🔐 Nexus AI Secure Login")

    # Toggle between modes
    auth_mode = st.radio("Authentication Mode", ["Standard Login", "Sign Up", "Passwordless (OTP)"], horizontal=True)

    if auth_mode == "Standard Login":
        with st.form("login_std"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In")

            if submitted:
                success, user = sign_in_with_password(email, password)
                if success:
                    st.session_state["authenticated"] = True
                    st.session_state["user_email"] = user.email
                    st.session_state["username"] = user.email.split("@")[0]  # Derive username
                    st.success("Welcome back!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Login failed: {user}")

    elif auth_mode == "Sign Up":
        with st.form("signup_form"):
            st.info("⚠️ Real email required for validation.")
            new_email = st.text_input("Email")
            new_pass = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create Account")

            if submitted:
                success, msg = sign_up_with_email(new_email, new_pass)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    elif auth_mode == "Passwordless (OTP)":
        if "otp_sent" not in st.session_state: st.session_state.otp_sent = False

        if not st.session_state.otp_sent:
            with st.form("otp_req"):
                email = st.text_input("Enter your Email")
                submitted = st.form_submit_button("Send OTP")
                if submitted:
                    success, msg = sign_in_with_otp(email)
                    if success:
                        st.session_state.otp_sent = True
                        st.session_state.otp_email = email
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        else:
            with st.form("otp_ver"):
                st.write(f"Enter code sent to: **{st.session_state.otp_email}**")
                token = st.text_input("6-Digit Code")
                submitted = st.form_submit_button("Verify & Login")
                if submitted:
                    success, user = verify_otp(st.session_state.otp_email, token)
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = user.email
                        st.session_state["username"] = user.email.split("@")[0]
                        st.rerun()
                    else:
                        st.error("Invalid Code")

            if st.button("Try different email"):
                st.session_state.otp_sent = False
                st.rerun()


def check_password():
    if st.session_state.get("authenticated", False):
        return True
    login_form()
    return False