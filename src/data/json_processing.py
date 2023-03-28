import pandas as pd
from typing import Dict, List
from datetime import datetime


class CreditReportAnalyzer:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self.df.set_index("application_id", inplace=True)

    def _get_acountrating_features(self, df_row: dict) -> Dict[str, float]:
        bad_accounts = sum([int(val) for key, val in df_row.items() if "bad" in key])
        good_accounts = sum([int(val) for key, val in df_row.items() if "good" in key])
        total_accounts = sum([int(val) for val in df_row.values()])
        ratio_good = good_accounts / total_accounts
        ratio_bad = bad_accounts / total_accounts

        feature_dict = {
            "bad_accounts": bad_accounts,
            "good_accounts": good_accounts,
            "acountrating_ratio_good": ratio_good,
            "acountrating_ratio_bad": ratio_bad,
        }

        return feature_dict

    def _get_telephonehistory_features(self, df_row: List[dict]) -> Dict[str, bool]:
        # FIXME df_row should be a List[Dict]
        hometel_nums = [d["hometelephonenumber"] for d in df_row]
        mobiletel_nums = [d["mobiletelephonenumber"] for d in df_row]
        change_homenumber = True if len(set(hometel_nums)) > 1 else False
        change_mobile = True if len(set(mobiletel_nums)) > 1 else False

        feature_dict = {
            "change_homenumber": change_homenumber,
            "change_mobile": change_mobile,
        }

        return feature_dict

    def _get_employmenthistory_features(
        self, df_row: List[dict]
    ) -> Dict[str, float]:  # TODO df_row is a list of Dict
        occupation = [d.get("occupation", "").upper() for d in df_row]
        # occupation is a list of all occupations in employement history
        # some of the are not in capital letters. Thats why I used upper to normalize
        number_of_employments = len(occupation)
        if any("PUBLIC" in occ for occ in occupation):
            employment_sector = "public"
        elif any("STUDENT" in occ for occ in occupation):
            employment_sector = "student"
        else:
            employment_sector = "private"
        # I iterate over this list and based on the 3 examples I provide 3 possible employement sectors
        feature_dict = {
            "number_of_employments": number_of_employments,
            "employment_sector": employment_sector,
        }

        return feature_dict

    def _get_creditaccountssummary_features(self, df_row: dict) -> Dict[str, float]:
        amountarrear = float(df_row["amountarrear"].replace(",", ""))
        # I converted to float because it was decimal values
        totaloutstandingdebt = float(df_row["totaloutstandingdebt"].replace(",", ""))
        totaldishonouredamount = float(
            df_row["totaldishonouredamount"].replace(",", "")
        )
        totaljudgementamount = float(df_row["totaljudgementamount"].replace(",", ""))

        ratio_arreas = amountarrear / totaloutstandingdebt
        ratio_dishonored = totaldishonouredamount / totaloutstandingdebt
        ratio_judgementamount = totaljudgementamount / totaloutstandingdebt

        feature_dict = {
            "ratio_arreas": ratio_arreas,
            "ratio_disnhored": ratio_dishonored,
            "ratio_judgementamount": ratio_judgementamount,
        }

        return feature_dict

    def _creditagreementssummary_feature(self, row: List[Dict]):
        # TODO row is a list of dictionaries. Should be df_row to make it more homogenous
        # FIXME remaining_to_original_outstanding_amount_ratio = current balance / opening balance
        count_open = 0
        ratios = []
        seasonings = []
        overdue_ratios = []
        for d in row:
            if d["accountstatus"] == "Open":
                count_open += 1
                openingbalanceamt = float(d["openingbalanceamt"].replace(",", ""))
                currentbalanceamt = float(d["currentbalanceamt"].replace(",", ""))
                try:
                    remaining_to_original_outstanding_amount_ratio = (
                        openingbalanceamt / currentbalanceamt
                    )
                except ZeroDivisionError:
                    remaining_to_original_outstanding_amount_ratio = 0

                ratios.append(remaining_to_original_outstanding_amount_ratio)
                seasoning = pd.to_datetime(d["lastupdateddate"]) - pd.to_datetime(
                    d["dateaccountopened"]
                )
                seasonings.append(seasoning)
                amountoverdue = float(d["amountoverdue"].replace(",", ""))
                try:
                    overdue_ratio = amountoverdue / currentbalanceamt
                except ZeroDivisionError:
                    overdue_ratio = 0
                overdue_ratios.append(overdue_ratio)
        if count_open > 0:
            mean_ratio = sum(ratios) / count_open
            mean_seasoning = sum(seasonings, pd.Timedelta(0)) / count_open
            mean_overdue_ratio = sum(overdue_ratios) / count_open
        else:
            mean_ratio = 0
            mean_seasoning = pd.Timedelta(0)
            mean_overdue_ratio = 0

        return {
            "open_status_loans": count_open,
            "remaining_to_original_outstanding_amount_ratio_mean_open_loans": mean_ratio,
            "seasoning_mean_open_loans": mean_seasoning,
            "overdue_ratio_mean_open_loans": mean_overdue_ratio,
        }

    def _personaldetailssummary_feature(self, row):
        birthdate = row["birthdate"]
        if birthdate != "":
            birthdate = datetime.strptime(birthdate, "%m/%d/%Y")
            age = (datetime.now() - birthdate).days // 365.25
        else:
            age = None
        return {"age": age}

    def get_features(self, application_id: str) -> Dict[str, float]:
        data = self.df.loc[application_id, "data"]["consumerfullcredit"]

        acountrating_features = self._get_acountrating_features(data["accountrating"])
        telephonehistory_features = self._get_telephonehistory_features(
            data["telephonehistory"]
        )
        employmenthistory_features = self._get_employmenthistory_features(
            data["employmenthistory"]
        )
        creditaccountssummary_features = self._get_creditaccountssummary_features(
            data["creditaccountsummary"]
        )
        creditagreementssummary_features = self._creditagreementssummary_feature(
            data["creditagreementsummary"]
        )
        personaldetailssummary_features = self._personaldetailssummary_feature(
            data["personaldetailssummary"]
        )

        feature_dict = {
            **acountrating_features,
            **telephonehistory_features,
            **employmenthistory_features,
            **creditaccountssummary_features,
            **creditagreementssummary_features,
            **personaldetailssummary_features,
        }

        return feature_dict
