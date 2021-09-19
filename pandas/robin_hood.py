import robin_stocks as rs
import pyotp

totp  = pyotp.TOTP("My2factorAppHere").now()
login = rs.robinhood.login('joshsmith@email.com','password')#, mfa_code=totp)
print("Current OTP:", totp)