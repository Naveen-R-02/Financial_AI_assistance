from transformers import pipeline
import random
import re
from rapidfuzz import process, fuzz

# --- Load the QA pipeline (same model you used) ---
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# --- Your BANKING_CONTEXT (unchanged) ---
# ----- Replace this dictionary in utils/ai_engine.py -----
BANKING_CONTEXT= {
    "debit card": (
        "A debit card is linked to your savings or current account and allows you to spend money directly from your balance. "
        "Steps to use a debit card: "
        "1. Insert the card at ATM/POS or tap if contactless. "
        "2. Enter your PIN for verification. "
        "3. Select withdrawal/payment option. "
        "4. Complete the transaction and collect cash/receipt."
    ),

    "credit card": (
        "A credit card allows you to borrow money up to a limit and repay later. "
        "Steps to use a credit card: "
        "1. Swipe, insert, or tap your credit card. "
        "2. Enter PIN or OTP for authentication. "
        "3. Pay the full bill or minimum amount before due date. "
        "4. Avoid late payments to prevent interest."
    ),

    "fixed deposit": (
        "A fixed deposit is a lump-sum investment for a fixed period at a higher interest rate. "
        "Steps to open a fixed deposit: "
        "1. Visit bank branch or open through net/mobile banking. "
        "2. Choose amount and tenure. "
        "3. Confirm interest payout frequency (monthly/quarterly/on maturity). "
        "4. Submit KYC if needed and complete payment."
    ),

    "saving account": (
        "A savings account is the basic bank account for storing money and earning small interest. "
        "Steps to open a savings account: "
        "1. Fill account opening form online or offline. "
        "2. Submit ID proof, address proof, and photos. "
        "3. Complete KYC verification. "
        "4. Deposit minimum balance if required."
    ),

    "current account": (
        "A current account is meant for businesses and frequent transactions. It usually doesn’t earn interest but provides higher withdrawal and deposit limits. "
        "Steps to open a current account: follow bank-specific KYC, submit business documents, and deposit initial funds as required."
    ),

    "interest": (
        "Interest is the cost of borrowing money or the reward for saving it. "
        "How it works (basic): "
        "1. For savings: bank credits interest periodically based on balance and rate. "
        "2. For loans: interest accrues on outstanding principal and is charged monthly/quarterly."
    ),

    "bank": (
        "A bank accepts deposits and provides credit, payments, and other financial services. "
        "How to use bank services: "
        "1. Open account with KYC documents. "
        "2. Use net/mobile banking or visit branch for services. "
        "3. Keep records and monitor statements regularly."
    ),

    "atm": (
        "An ATM allows cash withdrawal, balance enquiry, and mini statements. "
        "Steps to withdraw money from ATM: "
        "1. Insert your debit card. "
        "2. Enter your 4-digit PIN. "
        "3. Select ‘Withdraw Cash’ and amount. "
        "4. Collect cash and receipt; remember to take your card."
    ),

    "loan": (
        "A loan is borrowed money that must be repaid with interest. "
        "Steps to apply for a loan: "
        "1. Choose loan type (home, personal, education, vehicle). "
        "2. Prepare ID, address, income proofs, and property documents if secured. "
        "3. Submit application online or at bank branch. "
        "4. Bank verifies documents and credit score, then disburses on approval."
    ),

    "upi": (
        "UPI allows instant mobile-based money transfers. "
        "Steps to use UPI: "
        "1. Download and install a UPI app (Google Pay, PhonePe, BHIM). "
        "2. Register your mobile number linked to your bank. "
        "3. Create or enter UPI PIN. "
        "4. Scan QR code or enter UPI ID to send/receive money instantly."
    ),

    "net banking": (
        "Net banking allows online access to banking services from a browser. "
        "Steps to activate net banking: "
        "1. Visit your bank's website and choose ‘Register/Activate’. "
        "2. Enter account number, debit card details, and registered mobile. "
        "3. Set username and password; verify via OTP. "
        "4. Login and use services such as transfers and bill payment."
    ),

    "mobile banking": (
        "Mobile banking provides banking services via a smartphone app. "
        "Steps to use mobile banking: "
        "1. Install your bank's official mobile app. "
        "2. Register using your account number and mobile number. "
        "3. Complete OTP and MPIN setup. "
        "4. Access balance, transfers, mobile recharge, and more."
    ),

    "recurring deposit": (
        "A recurring deposit (RD) lets you save a fixed amount monthly and earn interest. "
        "Steps to open RD: "
        "1. Choose monthly installment and tenure. "
        "2. Set up auto-debit from your savings account. "
        "3. Pay installments monthly until maturity."
    ),

    "insurance": (
        "Insurance protects against financial loss (life, health, vehicle, crop). "
        "How to purchase insurance: "
        "1. Compare plans and premiums. "
        "2. Complete proposal form and KYC. "
        "3. Pay the premium online or at branch. "
        "4. Keep policy documents safe and claim when needed."
    ),

    "government schemes": (
        "Government schemes promote financial inclusion (e.g., Jan Dhan, PM Kisan). "
        "How to access: "
        "1. Check eligibility for the scheme. "
        "2. Apply online or at the designated office/branch. "
        "3. Provide required documents and complete verification."
    ),

    "financial literacy": (
        "Financial literacy means understanding saving, investing, borrowing, budgeting, and protecting your finances. "
        "How to improve: read guides, attend workshops, and practice budgeting."
    ),

    "digital payments": (
        "Digital payments are transactions done using phones, cards, or computers instead of cash. "
        "How to use: install trusted apps, link bank account/card, and authenticate transactions with UPI PIN/OTP."
    ),

    "banking security": (
        "Keep PINs, passwords, and OTPs private; use official apps and secure websites only. "
        "Basic security steps: "
        "1. Never share OTP/PIN. "
        "2. Enable two-factor authentication where possible. "
        "3. Keep apps updated and use strong passwords."
    ),

    "microfinance": (
        "Microfinance provides small loans to people without access to traditional banking, often for small businesses. "
        "How to apply: approach an MF institution, submit basic ID/market documents, and follow their lending process."
    ),

    "financial inclusion": (
        "Financial inclusion means everyone has access to affordable financial services. "
        "How to promote: open Jan Dhan accounts, conduct awareness drives, and provide low-cost services."
    ),

    "digital literacy": (
        "Digital literacy means having skills to safely use smartphones, banking apps, and UPI. "
        "How to learn: enroll in local training, watch tutorials, and practice on trusted apps."
    ),

    "apply a loan": (
        "To apply for a loan, submit identity, address, and income proofs and specify the loan purpose. "
        "Process summary: submit application → bank evaluates → sanction → disbursement."
    ),

    "credit score": (
        "A credit score measures creditworthiness based on repayment history. "
        "How to improve credit score: "
        "1. Pay EMIs and credit card bills on time. "
        "2. Keep balances low on credit cards. "
        "3. Avoid multiple loan applications in short time."
    ),

    "types of bank accounts": (
        "Common accounts: savings, current, fixed deposit, recurring deposit. "
        "How to choose: select based on need—savings for personal use, current for business."
    ),

    "types of loans": (
        "Common loans: personal, home, vehicle, education. "
        "How to choose: compare interest rates, tenure, and required documents."
    ),

    "overdraft": (
        "An overdraft lets you withdraw more than your account balance; interest applies on overdrawn amount. "
        "Steps to get an overdraft: "
        "1. Request overdraft facility at your bank (may need application). "
        "2. Provide income proof and agree on limit and charges. "
        "3. Use overdraft when needed and repay as agreed."
    ),

    "neft": (
        "NEFT (National Electronic Funds Transfer) transfers money in hourly batches and often works 24×7. "
        "Steps to send money via NEFT: "
        "1. Add beneficiary name, account number, and IFSC in net/mobile banking. "
        "2. Choose NEFT and enter amount. "
        "3. Confirm with OTP; funds settle in the next NEFT batch."
    ),

    "rtgs": (
        "RTGS (Real-Time Gross Settlement) transfers large amounts (min ₹2 lakh) instantly. "
        "Steps to use RTGS: "
        "1. Add beneficiary with correct bank & IFSC details. "
        "2. Choose RTGS in net/mobile banking and enter amount (≥ ₹2 lakh). "
        "3. Confirm and authenticate with OTP to transfer immediately."
    ),

    "imps": (
        "IMPS (Immediate Payment Service) enables instant 24×7 bank-to-bank transfers. "
        "Steps to send via IMPS: "
        "1. Select IMPS in your banking app. "
        "2. Use beneficiary account details or mobile+MMID. "
        "3. Enter amount and confirm with UPI/MPIN/OTP."
    ),

    "cheque": (
        "A cheque is a written instruction to your bank to pay a specified amount to someone. "
        "Steps to write/issue a cheque: "
        "1. Fill the date and payee name. "
        "2. Write the amount in numbers and words. "
        "3. Sign the cheque and hand it to the payee."
    ),

    "passbook": (
        "A passbook records deposits, withdrawals, and interest in your savings account. "
        "How to use passbook: "
        "1. Bring passbook to the branch for updates. "
        "2. Request teller to print recent transactions or use ATM passbook printer."
    ),

    "balance enquiry": (
        "Balance enquiry means checking available funds in your account. "
        "How to check balance: "
        "1. Use ATM, net banking, mobile app, or SMS service. "
        "2. For ATM: insert card and choose balance enquiry."
    ),

    "nominee": (
        "A nominee receives account funds if the account holder dies. "
        "Steps to add a nominee: "
        "1. Fill nominee section during account opening or submit a form. "
        "2. Provide nominee details and signature as required."
    ),

    "kcc": (
        "Kisan Credit Card (KCC) gives short-term credit to farmers for agricultural needs. "
        "How to apply for KCC: "
        "1. Visit bank with Aadhaar, land documents, and photo. "
        "2. Complete application and provide cropping/farming details. "
        "3. Bank sanctions limit based on eligibility."
    ),

    "mobile wallet": (
        "A mobile wallet stores digital money for quick payments. "
        "How to use a mobile wallet: "
        "1. Install the wallet app and register. "
        "2. Add money from bank/card or link UPI. "
        "3. Pay by scanning QR code or sending through app."
    ),

    "cif number": (
        "CIF (Customer Information File) is a unique identifier storing all customer account information. "
        "Where to find CIF: "
        "1. Check account opening documents or passbook. "
        "2. Ask customer care or branch to provide CIF."
    ),

    "moratorium": (
        "A moratorium is a temporary pause on loan repayments (e.g., during hardship). Interest may still accrue. "
        "How to request: "
        "1. Contact your bank and request moratorium relief. "
        "2. Provide supporting reasons/documents if required. "
        "3. Bank informs about interest and new repayment schedule."
    ),

    "emi": (
        "EMI (Equated Monthly Installment) is the fixed monthly payment to repay a loan. "
        "How to pay EMIs: "
        "1. Set up auto-debit from your bank account. "
        "2. Ensure sufficient balance on debit date. "
        "3. Check EMI schedule in loan account statements."
    ),

    "mutual fund": (
        "Mutual funds pool investor money to invest in stocks, bonds, or other assets. "
        "Steps to invest: "
        "1. Complete e-KYC (online KYC). "
        "2. Choose fund type and plan (Direct/Regular). "
        "3. Invest via SIP or lump sum and track performance."
    ),

    "upgrading kyc": (
        "KYC updation means submitting updated identity or address proofs to the bank. "
        "Steps to update KYC: "
        "1. Upload documents via net banking or visit branch with originals. "
        "2. Provide copies and complete biometric/OTP verification. "
        "3. Bank confirms update and notifies you."
    ),

    "ifsc code": (
        "IFSC is an 11-character code that uniquely identifies an Indian bank branch for electronic transfers. "
        "How to find IFSC: "
        "1. Check cheque book, passbook, or bank website. "
        "2. When adding a beneficiary for NEFT/RTGS/IMPS, enter the IFSC."
    ),

    "account statement": (
        "An account statement lists all transactions over a period. "
        "How to get a statement: "
        "1. Use net/mobile banking and choose statements. "
        "2. Select date range and download PDF or request branch to issue."
    ),

    "minimum balance": (
        "Minimum balance is the required amount to maintain to avoid penalties. "
        "How to manage: "
        "1. Check account rules for minimum balance. "
        "2. Keep required balance or upgrade/downgrade account type to avoid fees."
    ),

    "repo rate": (
        "Repo rate is the rate at which RBI lends to commercial banks, influencing loan costs. "
        "How it affects you: "
        "1. RBI changes repo rate during monetary policy meetings. "
        "2. Banks may raise or lower lending rates (EMI/loan rates) following repo changes."
    ),

    "reverse repo rate": (
        "Reverse repo is the rate at which RBI borrows from banks to absorb liquidity. "
        "How it is used: "
        "1. RBI changes reverse repo to control excess liquidity. "
        "2. Banks get interest on funds parked with RBI at this rate."
    ),

    "crr": (
        "CRR (Cash Reserve Ratio) is the share of deposits banks must keep with RBI in cash. "
        "How it affects lending: "
        "1. RBI increases CRR to reduce funds available for lending. "
        "2. Banks may reduce new loans or increase rates if CRR rises."
    ),

    "slr": (
        "SLR (Statutory Liquidity Ratio) is the portion of deposits banks must keep in safe assets. "
        "How it works: "
        "1. Banks maintain SLR in government securities or cash. "
        "2. Higher SLR reduces funds available for commercial lending."
    ),

    "inflation": (
        "Inflation is a sustained rise in prices over time, reducing purchasing power. "
        "How to protect savings: "
        "1. Invest in instruments that beat inflation (certain mutual funds, FDs with high rates). "
        "2. Monitor rates and rebalance investments periodically."
    ),

    "deflation": (
        "Deflation is a general decline in prices, often signaling weak demand. "
        "What to watch for: "
        "1. Falling prices can slow economic activity. "
        "2. Central bank may lower rates to stimulate demand."
    ),

    "collateral": (
        "Collateral is an asset pledged against a loan (property, gold). "
        "How to pledge collateral: "
        "1. Provide proof of ownership and valuation documents to the bank. "
        "2. Bank registers security interest and disburses loan against collateral."
    ),

    "secured loan": (
        "A secured loan requires collateral and typically offers lower interest. "
        "How to get a secured loan: "
        "1. Offer acceptable collateral (house, gold). "
        "2. Submit documents and valuation; bank approves and disburses at agreed terms."
    ),

    "unsecured loan": (
        "An unsecured loan has no collateral and depends on creditworthiness. "
        "How to apply: "
        "1. Provide ID, income proof, and bank statements. "
        "2. Bank assesses credit score and income before sanctioning the loan."
    ),

    "npa": (
        "NPA (Non-Performing Asset) is a loan where interest/principal is overdue for 90+ days. "
        "How banks handle NPA: "
        "1. Banks identify overdue account as NPA. "
        "2. They may restructure, recover, or write-off bad loans following RBI rules."
    ),

    "capital market": (
        "Capital markets trade long-term instruments like shares and bonds. "
        "How to participate: "
        "1. Open a Demat and trading account. "
        "2. Complete KYC and link bank account. "
        "3. Trade via brokers or investing platforms."
    ),

    "money market": (
        "Money market deals in short-term instruments like T-bills and commercial papers. "
        "How entities use it: "
        "1. Corporates and banks borrow short-term funds. "
        "2. Investors can buy short-term instruments for liquidity."
    ),

    "ledger balance": (
        "Ledger balance is the bank balance after all recorded transactions at day end. "
        "How it is calculated: "
        "1. Bank posts all finalized transactions to ledger. "
        "2. Ledger balance reflects end-of-day accounting."
    ),

    "available balance": (
        "Available balance is the amount you can immediately use; pending transactions may reduce it. "
        "How to check: "
        "1. Use net/mobile banking or ATM to view available balance. "
        "2. Remember pending debits (cheques, card payments) reduce availability."
    )
}
# ----- end BANKING_CONTEXT -----
def format_entry(value):
    """
    Accepts either:
      - dict with keys 'title','definition','how_to_use','example','tips'
      - list of strings (old style)
    Returns a nicely formatted multi-paragraph string.
    """
    if isinstance(value, dict):
        parts = []
        if value.get("title"):
            parts.append(f"{value['title']}.")
        if value.get("definition"):
            parts.append(f"{value['definition']}")
        if value.get("how_to_use"):
            parts.append(f"How to use / How it works: {value['how_to_use']}")
        if value.get("example"):
            parts.append(f"{value['example']}")
        if value.get("tips"):
            parts.append(f"Tip: {value['tips']}")
        # join with two newlines for clear paragraphs
        return "\n\n".join(parts)
    elif isinstance(value, (list, tuple)):
        # old-style: choose the longer string(s) and combine them
        # prefer longer strings and join up to 2 items
        sorted_items = sorted(value, key=lambda s: len(s), reverse=True)
        chosen = sorted_items[:2]  # combine two longest
        return "\n\n".join(chosen)
    else:
        return str(value)

# --- Helper: clean text for matching ---
def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or "").strip().lower())

# --- Main improved get_answer ---
def get_answer(query: str) -> str:
    """
    Improved keyword matching:
      1) whole-word / phrase exact match (prefers longer phrases)
      2) fuzzy match with RapidFuzz (prefers higher score and longer/specific keys)
      3) fallback to QA pipeline if no confident match

    Returns formatted KB entry via format_entry(...) when a KB key is selected.
    """
    if not query or not query.strip():
        return "Please ask a question about banking or finance."

    q = normalize(query)

    # 1) Exact whole-word / phrase match (prioritize longest match)
    exact_matches = []
    for key in BANKING_CONTEXT.keys():
        key_norm = normalize(key)
        # build whole-word/phrase regex (escape meta chars)
        pattern = r'\b' + re.escape(key_norm) + r'\b'
        if re.search(pattern, q):
            exact_matches.append(key)

    if exact_matches:
        # choose the longest (most specific) exact match
        best_key = max(exact_matches, key=lambda k: len(k))
        matched_value = BANKING_CONTEXT[best_key]
        return format_entry(matched_value)

    # 2) Fuzzy matching with RapidFuzz
    choices = list(BANKING_CONTEXT.keys())
    try:
        results = process.extract(q, choices, scorer=fuzz.WRatio, limit=3)
    except Exception:
        results = []

    if results:
        # results entries look like: (key, score, idx)
        best_key, best_score, _ = results[0]
        second_score = results[1][1] if len(results) > 1 else 0
        second_key = results[1][0] if len(results) > 1 else None

        # high-confidence match
        if best_score >= 85:
            return format_entry(BANKING_CONTEXT[best_key])

        # medium confidence: prefer the longer/more specific neighbor if scores are close
        if best_score >= 60:
            if second_key and abs(best_score - second_score) <= 8 and len(second_key) > len(best_key):
                return format_entry(BANKING_CONTEXT[second_key])
            return format_entry(BANKING_CONTEXT[best_key])

    # 3) Fallback to QA pipeline on whole KB (useful if KB doesn't match)
    try:
        # Flatten the KB into a single context string
        parts = []
        for v in BANKING_CONTEXT.values():
            if isinstance(v, dict):
                # join dict values (title, definition, how_to_use, example, tips)
                parts.append(" ".join([str(x) for x in v.values() if x]))
            elif isinstance(v, (list, tuple)):
                parts.append(" ".join([str(x) for x in v if x]))
            else:
                parts.append(str(v))
        context_text = " ".join(parts)

        res = qa_pipeline(question=query, context=context_text)
        ans = res.get("answer", "").strip()
        if ans:
            return ans
    except Exception:
        pass

    # final fallback
    return (
        "I'm not sure I have the exact answer for that, but I can tell you about "
        "banking topics like debit cards, credit cards, UPI, loans, insurance, or fixed deposits. "
        "Please ask about one of these!"
    )

