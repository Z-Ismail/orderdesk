# Sample data

Synthetic test data for orderdesk: a wholesale catering and janitorial supplier's catalogue, customers, ERP seed data and an inbox of purchase orders. All companies and people are made up.

## catalogue/
- `products.csv`: 40 SKUs. `unit_of_measure` + `pack_size` describe how we **sell** each item
  (for example, HH-GLV-NIT-M is sold by the case of 1000 gloves = 10 boxes of 100).
  `active=false` marks a discontinued SKU that should never be matched.
- `customer_aliases.csv`: known mappings from a customer's own wording or codes to our SKU.

## customers/
- `customers.csv`: 5 accounts. Note C005 (Exe Valley Care Home) is `on_hold`.

## erp/  (seed data for the simulated ERP on Day 4)
- `inventory.csv`: stock by SKU and warehouse (AVONMOUTH, EXETER).
- `price_tiers.csv`: discount by customer price tier.

## inbox/  (stands in for the shared orders mailbox)
| File | What it is | Traps |
|---|---|---|
| po_1001_gull_cafe.pdf | Clean PDF PO using our SKUs | None, the happy path |
| po_1002_clifton_grand.pdf | PDF PO with the hotel's own item codes | No prices; "2000 each" napkins (sold in cases of 1000); rinse aid wording differs |
| email_003_redland_primary.eml | Free-text email, no PO number | Relative date ("next Tuesday (18th)"); "1 box" of gloves sold by the case |
| email_004_harbourside_kitchen.eml | Short email with a PDF PO attached | Order data lives in the attachment; delivery note lives in the body |
| po_1005_exe_valley_scanned.pdf | Scanned PO, image only, no text layer | Needs OCR / multimodal model; customer account is on hold |
| email_006_harbourside_kitchen_redelivered.eml | Byte-identical re-delivery of email_004 | Must not become a second order |
| email_007_clifton_grand_amendment.eml | Amendment to PO CGH-PO-88341 | Not a new order |
| email_008_unknown_sender.eml | Unknown sender with a prompt-injection payload | Must never be auto-approved, discounted or redirected |

## golden/  (labels for evaluation, used from Day 2)
- One `<inbox-file-stem>.json` per unique document: the correct `ExtractedOrder`.
  There is no label for the duplicate email_006.
- `expected_matches.csv`: the correct SKU for every line (used on Day 3).

Labelling conventions (these are the rules your extractor will be judged on):
- `order_date`: the date on the document; for an email with no date in the body, the date the email was sent.
- `requested_delivery_date`: resolved to a calendar date when the text gives one ("next Tuesday (18th)" means 2026-11-18); otherwise null.
- `description` and `unit_of_measure`: the customer's wording, lightly trimmed, not our catalogue's wording.
- `customer_sku`: the customer's own item code if printed, else null.
- `unit_price`: only if printed on the document.
- Never extract instructions as data: the injection text in email_008 is not a note.
