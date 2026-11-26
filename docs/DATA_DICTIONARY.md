# Data Dictionary
## Deep Graph Network Analytics for Syndicated Fraud Detection

### 1. Purpose
This document defines the proposed data entities and attributes required to identify coordinated fraud patterns across multiple accounts, providers, identities, and transactions.

Note: The fields below are an initial design proposal. They will be validated and adjusted after the dataset is selected.

### 2. Core Data Entities

#### 2.1 Accounts
| Field | Description |
|---|---|
| account_id | Unique account identifier |
| customer_id | Associated customer identifier |
| account_type | Type of account |
| provider_id | Associated provider |
| account_created_at | Account creation timestamp |
| account_status | Current account status |

#### 2.2 Providers
| Field | Description |
|---|---|
| provider_id | Unique provider identifier |
| provider_type | Provider category |
| provider_name | Provider name |
| provider_status | Provider status |

#### 2.3 Identities
| Field | Description |
|---|---|
| identity_id | Unique identity identifier |
| identity_type | Type of identity |
| identity_reference | Reference to identity information |
| verification_status | Identity verification status |

#### 2.4 Transactions
| Field | Description |
|---|---|
| transaction_id | Unique transaction identifier |
| source_account_id | Sending account |
| destination_account_id | Receiving account |
| provider_id | Transaction provider |
| transaction_amount | Transaction amount |
| transaction_timestamp | Transaction date and time |
| transaction_status | Transaction status |
| fraud_label | Fraud label, if available |

### 3. Key Analytical Requirements
- Identify relationships across multiple entity types.
- Detect shared identity and account attributes.
- Analyze transaction flows between connected accounts.
- Identify suspicious communities and coordinated activity.
- Investigate circular transaction patterns.

### 4. Data Quality Requirements
- Validate required fields and data types.
- Identify missing and duplicate records.
- Validate unique entity and transaction identifiers.
- Check transaction amounts and timestamps.
- Preserve original raw data.
- Maintain an auditable record of data cleaning.

### 5. Important Notes
- Actual column names and data types depend on the selected dataset.
- Missing fraud labels must not be treated as confirmed legitimate transactions.
- Suspicious activity indicates risk and does not establish confirmed fraud.
