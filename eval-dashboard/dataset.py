"""Curated 22-question evaluation set for the Finance Insight Assistant.

Each entry has three parallel fields:
- queries[i]              : the user question
- expected_responses[i]   : the gold reference answer (used by Faithfulness /
                            FactualCorrectness as the ground truth)
- expected_supporting_docs[i] : a short representative supporting passage,
                            useful for spot checks (not consumed by RAGAS in
                            the current pipeline but kept for traceability).

Lifted from evaluation/Evaluations.ipynb so the notebook and the dashboard
share one source of truth.
"""

queries = [
    "What factors contribute to the volatility of Abiomed's common stock price?",
    "On which stock exchange is Abiomed listed, and what is its ticker symbol?",
    "Does Abiomed rely heavily on any single customer for its revenue?",
    "How did ACNB Corporation's dividend payments change during 2020?",
    "What is the principal component of ACNB Corporation's net income, and what factors influence it?",
    "What financial services does Russell Insurance Group, Inc. provide?",
    "Why might Adobe's stock price fluctuate even if the company is performing well?",
    "Does Adobe offer stock purchase benefits to its employees?",
    "What financial challenges has Applied DNA Sciences faced in recent years?",
    "Is there any risk related to the listing of Applied DNA Sciences' stock on NASDAQ?",
    "How do fluctuations in oil and gas prices affect an energy company's financial performance?",
    "What risks do Abraxas Petroleum shareholders face regarding stock dilution?",
    "Has Abraxas Petroleum paid dividends on its common stock, and how has its stock price performed in recent years?",
    "What was Ames National Corporation's dividend yield in 2022, and how does it compare to its stock performance?",
    "What were the key financial performance indicators for Ames National Corporation in 2022?",
    "in 2015, what was the reported sales value for Aceto that year and how much increase from the last year it represented?",
    "What were the primary drivers behind the increase in Aceto's Selling, General and Administrative (SG&A) expenses in 2016?",
    "how much did Aceto's gross profit increase in 2015 compared to the previous year?",
    "What are the primary methods Aceto uses to market and distribute its products, and where are these activities concentrated?",
    "Who are Aceto's largest customers, and what percentage of their revenue do these customers represent?",
    "How has the COVID-19 pandemic affected the company in terms of operations, finances, and stock price of Apple?",
    "How did Apple generate and use cash in its operating, investing, and financing activities during 2020?",
]

expected_responses = [
    "The volatility of Abiomed's common stock price is influenced by several factors, including variations in quarterly operational results, regulatory approvals, FDA announcements, customer reviews, new product introductions by Abiomed or its competitors, strategic alliances or acquisitions, changes in healthcare policies, and shifts in market conditions or economic factors.",
    "Abiomed is listed on the NASDAQ Global Select Market under the ticker symbol ABMD.",
    "Abiomed does not rely on any single customer for its revenue. No customer accounted for more than 10% of the company's total revenues in fiscal years 2019, 2018, or 2017.",
    "ACNB Corporation increased its total dividend payments in 2020 despite the pandemic's challenges. The annual dividend per share rose slightly compared to 2019, and the overall dividends paid to shareholders grew by 25.5%, influenced in part by the acquisition of Frederick County Bancorp, Inc.",
    "The principal component of ACNB Corporation's net income is net interest income, which comes from the interest earned on loans and investments, minus the interest paid on deposits and borrowings. This income is influenced by changes in interest rates, the volume and composition of interest-earning assets and liabilities, local economic conditions, and competitive market dynamics.",
    "Russell Insurance Group, Inc. offers a broad range of property, casualty, health, life, and disability insurance services to both personal and commercial clients.",
    "Adobe's stock price may fluctuate due to broader market volatility or shifts in investor confidence in the technology sector, even if the company's operations are strong. Such fluctuations are often unrelated to Adobe's financial performance and can expose the company to costly legal actions like securities class litigation.",
    "Yes, Adobe offers an Employee Stock Purchase Plan (ESPP) that allows eligible employees to buy company stock at a 15% discount. The stock can be purchased at the lower market price from either the start of the offering period or the end of the purchase period, making it a financially advantageous benefit for employees.",
    "Applied DNA Sciences has reported consistent net losses, including $8.6 million in 2019 and $11.7 million in 2018. These losses were primarily due to high operational, administrative, and research costs related to expanding operations and developing technologies. Continued losses may hinder the company's ability to secure additional financing, potentially affecting its long-term viability.",
    "Yes, while Applied DNA Sciences' common stock is listed on the NASDAQ Capital Market under the symbol APDN, there is no guarantee that it will remain listed or that sufficient liquidity will be available for shareholders. The company's warrants, previously listed under 'APDNW,' expired in November 2019, further impacting investor options.",
    "Fluctuations in oil and gas prices can significantly affect an energy company's revenue, profitability, and cash flow. When prices decrease, the company may face reduced revenues and lower profitability, which can hinder growth and financial stability. These price changes are influenced by factors like seasonality, economic conditions, foreign imports, and political events.",
    "Abraxas Petroleum shareholders face the risk of dilution if the company issues additional shares of common stock. This dilution can reduce the ownership percentage of existing shareholders and create downward pressure on the stock price. The perception or actual issuance of additional shares for capital raising can adversely affect stock value.",
    "Abraxas Petroleum has not paid any cash dividends on its common stock. The company's stock price has shown significant fluctuations from 2015 to early 2017, with highs reaching $3.98 in the second quarter of 2015 and lows dropping to $0.65 in the first quarter of 2016.",
    "In 2022, Ames National Corporation had a dividend yield of 4.57%, based on total dividends of $1.08 per share. The company maintained a quarterly dividend of $0.27, and its stock closed the year at $23.61, reflecting steady dividend payouts relative to its share price.",
    "In 2022, Ames National Corporation reported a net income of $19.3 million and total assets of $2.1 billion. The company declared dividends of $1.08 per share, with a return on average assets of 0.90% and a return on average equity of 11.43%. The closing stock price at the end of the year was $23.61.",
    "we are reporting net sales of $558,524 for the year ended June 30, 2015, which represents a 2.1% increase from the $546,951 reported in the comparable prior year",
    "The increase in SG&A is primarily due to increased stock-based compensation expense of $2,182. SG&A for the current year also included $1,213 of transaction costs related to a potential acquisition of a target company that we evaluated during the year but ultimately determined not to pursue, as well as $1,313 environmental remediation charge related to Arsynco",
    "gross profit increased $20,731 or 18.1% to $135,434 (24% of net sales) for the year ended June 30, 2015, as compared to $114,703 (22.5% of net sales) for the prior year.",
    "Aceto's marketing and distribution efforts are primarily concentrated in the United States and Europe. However, they've been expanding into other regions like Japan, China, other parts of Asia, Latin America, and the Middle East.  They use a variety of approaches, including their own affiliates, acquiring existing businesses or product rights, and collaborating with third parties.  In the US, they mainly sell through pharmaceutical wholesale distributors, while outside the US, they sell directly to healthcare providers and/or distributors, depending on local practices. They also use direct-to-consumer channels like print, television, and online media for some products.",
    "Aceto's three largest customers are pharmaceutical wholesale distributors: McKesson Corporation, AmerisourceBergen Corporation, and Cardinal Health, Inc.  Each of these individually accounts for more than 10% of Aceto's total revenue.  Combined, these three wholesalers represent a significant portion of Aceto's business, accounting for 82% of worldwide gross revenues in both 2022 and 2021, and 83% in 2020.",
    "The COVID-19 pandemic and the measures taken by many countries in response have adversely affected and could in the future materially adversely impact the Company's business, results of operations, financial condition and stock price. Following the initial outbreak of the virus, the Company experienced disruptions to its manufacturing, supply chain and logistical services provided by outsourcing partners, resulting in temporary iPhone supply shortages that affected sales worldwide. During the course of the pandemic, the Company's retail stores, as well as channel partner points of sale, have been temporarily closed at various times. Additional future impacts on the Company may include, but are not limited to, material adverse effects on: demand for the Company's products and services; the Company's supply chain and sales and distribution channels; the Company's ability to execute its strategic plans; and the Company's profitability and cost structure.",
    "During 2020, cash generated by operating activities of $80.7 billion was a result of $57.4 billion of net income, non-cash adjustments to net income of $17.6 billion and an increase in the net change in operating assets and liabilities of $5.7 billion. Cash used in investing activities of $4.3 billion during 2020 consisted primarily of cash used to acquire property, plant and equipment of $7.3 billion and cash paid for business acquisitions, net of cash acquired, of $1.5 billion, partially offset by proceeds from maturities and sales of marketable securities, net of purchases, of $5.5 billion. Cash used in financing activities of $86.8 billion during 2020 consisted primarily of cash used to repurchase common stock of $72.4 billion, cash used to pay dividends and dividend equivalents of $14.1 billion, cash used to repay or redeem term debt of $12.6 billion and net repayments of commercial paper of $1.0 billion, partially offset by net proceeds from the issuance of term debt of $16.1 billion.",
]

# Sanity check at import time so a missing entry fails loudly during boot
# rather than silently producing a partial eval.
assert len(queries) == len(expected_responses), (
    "queries and expected_responses must be parallel — got "
    f"{len(queries)} queries and {len(expected_responses)} expected_responses"
)

NUM_QUESTIONS = len(queries)
