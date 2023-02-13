import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
from modules.transactions.tree_manager import TreeManager
from sklearn.cluster import DBSCAN
import re
from functools import reduce
from config import *


class FrozenGenerator:
    def __init__(self, s_reference_date, i_days, d_offer_info):
        self.s_reference_date = s_reference_date
        self.i_days = i_days
        self.d_offer_info = d_offer_info
        self.treeManager = TreeManager()

    def update_dict_value(self, dictionary, keys, new_values):
        for k, v in zip(keys, new_values):
            dictionary[k] = v
        return dictionary

    def search_regex(self, s_text, s_pattern):
        regex_result = re.search(s_pattern, s_text)
        if regex_result is not None:
            ini, end = regex_result.span()
            return s_text[ini:end]
        else:
            return None

    def get_incomes(self, df_transactions):
        """
        :param df_transactions: dataframe of transactions of one or more userids
        :return: a pandas dataframe consisting of all incomes (as columns) and all userids (as rows)
        """

        def get_cluster_incomes(df, income_type, eps=5, max_days=35):
            def _dbscan(data, income_type, eps=eps, max_days=max_days):
                days_since = data.days_difference.values
                if np.min(days_since) < max_days:
                    km = DBSCAN(eps=eps, min_samples=1).fit(days_since.reshape(-1, 1))
                    data = pd.DataFrame(
                        data={income_type: data.quantity.values, "cluster": km.labels_}
                    )
                else:
                    data = pd.DataFrame(data={income_type: [0.0], "cluster": [0]})
                return data

            try:
                df_ = (
                    df.groupby("userid")
                    .apply(lambda df: _dbscan(df, income_type))
                    .groupby(["userid", "cluster"])
                    .agg({income_type: "sum"})
                    .reset_index()
                    .groupby(["userid"])
                    .agg({income_type: "median"})
                    .reset_index()
                )
                return df_
            except:
                raise e

        def get_income_value(df, income_type):
            df_ = df.loc[
                (
                    df["category"].isin(
                        d_incl_excl["income"][income_type]["cat_inclusions"]
                    )
                    & df["grupotipo"].isin(
                        d_incl_excl["income"][income_type]["gt_inclusions"]
                    )
                )
                & (df.quantity >= d_incl_excl["income"][income_type]["min_quantity"])
            ].copy()
            try:
                df_ = get_cluster_incomes(df_, income_type)
            except:
                df_ = pd.DataFrame({"userid": [df.userid.iloc[0]], income_type: [0]})
            return df_

        df_transactions_ = df_transactions.loc[
            (df_transactions["quantity"] > 0)
            & (df_transactions["days_difference"] < self.i_days)
        ].copy()

        l_df_incomes = list(
            map(
                lambda inc_type: get_income_value(df_transactions_, inc_type),
                ["payroll", "pension", "unemployment"],
            )
        )
        df_incomes = reduce(
            lambda left, right: pd.merge(left, right, on="userid", how="left"),
            l_df_incomes,
        )

        # Incomes tree
        df_ts_incomes = self.treeManager.get_income_tree(
            df_transactions, i_days=(3 * 31)
        ).get_df()

        try:
            # Minimun conditions to consider the incomes
            income_type = "others"
            r_out = """
                compra|tarjet|hucha|ahorro|bizum|factoring|confirming|tpv|remesa|
                recarga|devol|ab movil tfr|trustly group ab|csh drw pay|paypal|
                traspaso propio
                """
            r_out = "(" + re.sub("(\n)( *)", "", r"%s" % (r_out)) + ")"
            cond_others_1 = (
                (
                    (
                        df_ts_incomes["category"].isin(
                            d_incl_excl["income"][income_type]["cat_inclusions"]
                        )
                        & df_ts_incomes["grupotipo"].isin(
                            d_incl_excl["income"][income_type]["gt_inclusions"]
                        )
                    )
                    | (
                        df_ts_incomes["category"].isin(["I0101", "I0103", "I0106"])
                        & df_ts_incomes["grupotipo"].isin(
                            d_incl_excl["income"][income_type]["gt_inclusions"]
                        )
                    )
                    | (
                        df_ts_incomes["category"].isin(
                            d_incl_excl["income"][income_type]["cat_inclusions"]
                        )
                        & df_ts_incomes["grupotipo"].isin(
                            [
                                "INGRESOS_NOMINA",
                                "INGRESOS_PENSION",
                                "INGRESOS_AYUDAS",
                                "INGRESOS_PARO",
                            ]
                        )
                    )
                )
                & (~df_ts_incomes["texto"].str.contains(r_out, case=False))
                & (df_ts_incomes["periodo"].between(27, 34))
                & (df_ts_incomes["cercania"] > 93)
                & (df_ts_incomes["estabilidad"] >= 30)
            )
            # Others incomes calculation
            # Si en el concepto aparece "ingreso minimo vital"
            ingreso_minimo = max(
                df_ts_incomes["texto"].str.contains("ingreso minimo vital")
            )
            # Si en el concepto aparece "prestamo", "coche", "hipoteca", "pres" o "gastos"
            cond_help = df_ts_incomes["texto"].str.contains(
                "(coche|hipoteca|pres|ayud|gasto)"
            )
            cond_others_2 = cond_help
            # Si en el concepto aparece "alquiler", "piso", etc y dice que tiene vivienda propia o tiene hipoteca
            cond_rent = df_ts_incomes["texto"].str.contains(
                "(alquil|piso|lloguer|garaje|renta calle)"
            )
            if self.d_offer_info["tipo_de_residencia"] in [
                "Piso en propiedad",
                "Casa en propiedad",
            ]:
                cond_others_2 |= cond_rent
            # Si dice que es autónomo
            if self.d_offer_info["situacion_laboral"] == "Autónomo":
                cond_others_2 |= ~cond_rent
            # Si dice que no es autónomo
            if self.d_offer_info["situacion_laboral"] not in [
                "Autónomo",
                "Desconocido",
            ]:
                # En el concepto aparece mutua y la cantidad es menor de 500€
                cond_others_2 |= (df_ts_incomes["texto"].str.contains("(mutua)")) & (
                    df_ts_incomes["cantidad"] < 500
                )
                # Ya tiene nómina
                if df_incomes["payroll"].iloc[0] > 0:
                    cond_others_2 |= ~cond_rent
                else:  # No tiene nómina y la estabilidad es menor de 70
                    cond_others_2 |= (df_ts_incomes["estabilidad"] < 70) & (~cond_rent)
            df_others = (
                df_ts_incomes.loc[cond_others_1 & cond_others_2]
                .groupby("userid")
                .agg({"cantidad": "sum"})
                .reset_index()
                .rename(columns={"cantidad": "others"})
            )
            df_incomes = df_incomes.merge(df_others, on="userid", how="left")
            # Si no es autónomo, pensionista ni desempleado
            if self.d_offer_info["situacion_laboral"] not in [
                "Autónomo",
                "Desempleado",
                "Pensionista",
                "Desconocido",
            ]:
                # No tiene ingresos de nómina y los ingresos son estables
                if df_incomes["payroll"].iloc[0] == 0:
                    cond_payroll = (
                        (df_ts_incomes["cantidad"].between(400, 6000))
                        & (df_ts_incomes["estabilidad"] >= 70)
                        & (~cond_rent)
                        & (~cond_help)
                    )
                    df_payroll = (
                        df_ts_incomes.loc[cond_others_1 & cond_payroll]
                        .groupby("userid")
                        .agg({"cantidad": "sum"})
                        .reset_index()
                        .rename(columns={"cantidad": "payroll_2"})
                    )
                    if len(df_payroll):
                        df_incomes["flag_new_incomes_payroll"] = True
                    else:
                        df_incomes["flag_new_incomes_payroll"] = False
                    df_incomes = df_incomes.merge(df_payroll, on="userid", how="left")
                    df_incomes["payroll"] += df_incomes["payroll_2"].fillna(0)
                    cols = ["grupotipo", "category"]
                    values = ["INGRESOS_NOMINA", "I0101"]
                    df_ts_incomes.loc[cond_others_1 & cond_payroll, cols] = values
                    df_ts_incomes.loc[
                        cond_others_1 & cond_payroll, ["transacciones"]
                    ] = df_ts_incomes.loc[cond_others_1 & cond_payroll].apply(
                        lambda r: [
                            self.update_dict_value(d, cols, values)
                            for d in r.transacciones
                        ],
                        axis=1,
                    )
        except Exception as e:
            logger.exception(e)

        # Non-recurrent incomes median
        df_ts_non_recurrent_incomes = df_ts_incomes.query("tipo == 'NO_RECURRENTE'")
        if len(df_ts_non_recurrent_incomes):
            df_non_recurrent_incomes = pd.concat(
                [pd.DataFrame(l) for l in df_ts_non_recurrent_incomes["transacciones"]]
            )
            df_non_recurrent_incomes["month"] = pd.to_datetime(
                df_non_recurrent_incomes["valuedate"]
            ).dt.month
            df_non_recurrent_incomes_median = (
                df_non_recurrent_incomes.groupby(["userid", "month"])
                .agg({"quantity": "sum"})
                .groupby("userid")
                .agg({"quantity": "median"})
                .rename(columns={"quantity": "non_recurrent_income"})
                .reset_index()
            )
            df_incomes = df_incomes.merge(
                df_non_recurrent_incomes_median, on="userid", how="left"
            )

        # Adding incomes timelines to response object
        df_ts_incomes["texto"] = df_ts_incomes["texto"].str.capitalize()
        df_ts_incomes = (
            df_ts_incomes[
                [
                    "userid",
                    "grupotipo",
                    "category",
                    "texto",
                    "cantidad",
                    "estabilidad",
                    "recurrencia",
                    "cercania",
                    "tipo",
                    "transacciones",
                ]
            ]
            .groupby("userid")
            .apply(lambda df: df.to_dict("records"))
            .reset_index(name=f"ts_incomes")
        )
        df_incomes = df_incomes.merge(df_ts_incomes, on="userid", how="left")

        # Monthly mean incomes
        all_gt_incomes = [
            c
            for k in d_incl_excl["income"].keys()
            for c in d_incl_excl["income"][k]["gt_inclusions"]
        ]
        all_cat_incomes = [
            c
            for k in d_incl_excl["income"].keys()
            for c in d_incl_excl["income"][k]["cat_inclusions"]
        ]
        df_all_incomes = df_transactions_.loc[
            (df_transactions_["days_difference"] <= 80)
            & (df_transactions_["quantity"] >= 500)
            & (df_transactions_["quantity"] <= 10000)
            & (df_transactions_["grupotipo"].isin(all_gt_incomes))
            & (df_transactions_["category"].isin(all_cat_incomes))
        ].copy()
        df_all_incomes = (
            df_all_incomes.groupby(by="userid")
            .agg({"quantity": "sum"})
            .reset_index()
            .rename(columns={"quantity": "all_incomes"})
        )
        df_all_incomes["mean_all_incomes"] = df_all_incomes["all_incomes"] / 3
        df_incomes = df_incomes.merge(df_all_incomes, on="userid", how="left")
        df_incomes["flag_ingreso_minimo"] = ingreso_minimo
        # Si hay más de dos INGRESOS_NOMINA
        # contienen textos diferentes
        # y la cantidad es mayor a 400
        df_incomes["flag_double_income"] = df_incomes["ts_incomes"].apply(
            lambda x: True
            if (
                sum(
                    [
                        1
                        if i["grupotipo"] == "INGRESOS_NOMINA" and i["cantidad"] >= 400
                        else 0
                        for i in x
                    ]
                )
                >= 2
                and pd.Series(
                    [i["texto"] for i in x if i["grupotipo"] == "INGRESOS_NOMINA"]
                ).is_unique
            )
            else False
        )
        df_incomes["flag_income_zero"] = np.where(
            (df_incomes["payroll"] == 0) & (df_incomes["pension"] == 0), True, False
        )

        return df_incomes

    def get_indebtedness(self, df_transactions, df_cards, df_loans):
        def get_efc_loans(df_transactions, df_loans):
            def build_dict(r):
                return dict(zip(r.tipo, r.info_list))

            def set_deduction_based_on_tx(df_loans, df_ts_efcs):
                try:
                    # Getting loans with unknown deduction
                    loans_cols = [
                        "userid",
                        "system_bankid",
                        "bank_name",
                        "initial_balance",
                        "balance",
                        "value_rate",
                        "loan_type",
                        "account_id",
                        "begin_date",
                        "end_date",
                        "webAlias",
                    ]
                    df_loans_ = (
                        df_loans.reset_index()
                        .rename(columns={"index": "aux_index"})
                        .query("deduction == 0")
                        .sort_values("initial_balance", ascending=False)
                        .reset_index(drop=True)[["aux_index"] + loans_cols]
                    )
                    # Getting instalments from efcs_ts
                    df_ts_efcs_ = df_ts_efcs.copy()
                    df_ts_efcs_["cantidad"] = df_ts_efcs_["cantidad"].abs()
                    df_efcs_ = (
                        df_ts_efcs_.loc[
                            (df_ts_efcs_.commerce_name == "")
                            & (~df_ts_efcs_.cantidad.isin(df_loans.deduction.tolist()))
                            & (
                                df_ts_efcs_.system_bankid.isin(
                                    df_loans_.system_bankid.tolist()
                                )
                            )
                        ]
                        .sort_values("cantidad", ascending=False)
                        .reset_index()
                        .rename(columns={"cantidad": "deduction"})[["deduction"]]
                    )
                    # Concatenating both dataframes
                    min_len = min(len(df_efcs_), len(df_loans_))
                    df_loans_ = pd.concat(
                        [df_efcs_.iloc[:min_len], df_loans_.iloc[:min_len]], axis=1
                    )
                    df_loans = pd.concat(
                        [
                            df_loans.loc[
                                ~df_loans.index.isin(df_loans_.aux_index.tolist()),
                                loans_cols + ["deduction"],
                            ],
                            df_loans_[loans_cols + ["deduction"]],
                        ]
                    ).reset_index(drop=True)
                    return df_loans
                except:
                    return df_loans

            def check_duplicated_loans(df_ts_efcs, df_loans, thr_deduction=0.05):
                try:
                    df_ts_efcs_ = df_ts_efcs.copy()
                    df_ts_efcs_["cantidad"] = df_ts_efcs_["cantidad"].abs()
                    df_ts_efcs_ = df_ts_efcs_.reset_index().rename(
                        columns={"index": "index_efcs"}
                    )
                    df_loans_ = df_loans.copy()
                    df_loans_ = (
                        df_loans[
                            ["system_bankid", "deduction", "bank_name", "loan_type"]
                        ]
                        .reset_index()
                        .rename(columns={"index": "index_loan"})
                    )
                    df_ts_efcs_in_loan = df_ts_efcs_.query(
                        'commerce_name.isin(["", "banco cetelem", "cajamar consumo"])'
                    ).merge(df_loans_, on=["system_bankid"])
                    df_ts_efcs_in_loan["relative_diff"] = df_ts_efcs_in_loan.apply(
                        lambda r: abs(r.deduction - r.cantidad)
                        / max(r.deduction, 0.01),
                        axis=1,
                    )
                    df_ts_efcs_in_loan["in_loan_table"] = df_ts_efcs_in_loan.apply(
                        lambda r: r.relative_diff <= thr_deduction, axis=1
                    )
                    df_ts_efcs_in_loan = (
                        df_ts_efcs_in_loan.query("in_loan_table == True")
                        .sort_values(["index_loan", "commerce_name", "relative_diff"])
                        .drop_duplicates("index_loan")
                        .sort_values(["commerce_name", "relative_diff"])
                        .drop_duplicates("index_efcs")
                    )
                    df_ts_efcs_ = pd.merge(
                        df_ts_efcs_,
                        df_ts_efcs_in_loan[
                            ["index_efcs", "in_loan_table", "bank_name", "loan_type"]
                        ],
                        on="index_efcs",
                        how="left",
                    )
                    df_ts_efcs_["in_loan_table"].fillna(False, inplace=True)
                    df_ts_efcs_[["bank_name", "loan_type"]] = df_ts_efcs_[
                        ["bank_name", "loan_type"]
                    ].fillna("")
                    df_ts_efcs_["commerce_name"] = df_ts_efcs_.apply(
                        lambda r: r.bank_name if r.in_loan_table else r.commerce_name,
                        axis=1,
                    )
                    df_ts_efcs_["grupotipo"] = df_ts_efcs_.apply(
                        lambda r: "PRESTAMOS_CUOTA_HIPOTECA"
                        if r.loan_type == "HIPOTECA"
                        else r.grupotipo,
                        axis=1,
                    )
                    df_ts_efcs_["category"] = df_ts_efcs_.apply(
                        lambda r: "G0701" if r.loan_type == "HIPOTECA" else r.category,
                        axis=1,
                    )
                    df_ts_efcs_["transacciones"] = df_ts_efcs_.apply(
                        lambda r: [
                            self.update_dict_value(
                                d,
                                ["grupotipo", "category"],
                                ["PRESTAMOS_CUOTA_HIPOTECA", "G0701"],
                            )
                            for d in r.transacciones
                        ]
                        if r.loan_type == "HIPOTECA"
                        else r.transacciones,
                        axis=1,
                    )
                    df_ts_efcs_["cantidad"] = -df_ts_efcs_["cantidad"]
                    df_ts_efcs_.drop(
                        columns=["index_efcs", "bank_name", "loan_type"], inplace=True
                    )
                    return df_ts_efcs_.reset_index(drop=True)
                except:
                    return df_ts_efcs

            df_ts_efcs = self.treeManager.get_indebtedness_tree(
                df_transactions
            ).get_df()
            df_loans = set_deduction_based_on_tx(df_loans, df_ts_efcs)
            # Loan de-duplication between transactional and loan table
            df_ts_efcs = check_duplicated_loans(df_ts_efcs, df_loans)
            try:
                df_ts_efcs_ = df_ts_efcs.query("in_loan_table == False")
            except:
                df_ts_efcs_ = df_ts_efcs
            # Exclude "union de creditos inmobiliarios" from efcs (it's always mortgage)
            # and include it in loan table
            df_uci = df_ts_efcs_.query("commerce_id == 153066").copy()
            if len(df_uci):
                df_ts_efcs_ = df_ts_efcs_.query("commerce_id != 153066")
                df_uci = df_uci[["userid", "commerce_name", "cantidad"]].rename(
                    columns={"commerce_name": "bank_name", "cantidad": "deduction"}
                )
                df_uci["loan_type"] = "HIPOTECA"
                df_loans = pd.concat([df_loans, df_uci]).fillna(0)
            try:
                # Partner info
                df_ts_efcs_["info_list"] = df_ts_efcs_.apply(
                    lambda x: {
                        "endBankId": x["commerce_name"].capitalize(),
                        "endBankCuot": np.abs(x["cantidad"]),
                        "detectionSource": x["detection_source"],
                    },
                    axis=1,
                )
                df_efcs = (
                    df_ts_efcs_.groupby(["userid", "tipo"])["info_list"]
                    .apply(list)
                    .reset_index()
                )
                df_efcs = (
                    df_efcs.groupby(["userid"])
                    .apply(build_dict)
                    .reset_index(name="efc_dict")
                )
            except:
                df_efcs = pd.DataFrame({"userid": df_transactions.userid.unique()})
            try:
                # Adding indebtness timelines to response object
                df_ts_efcs["commerce_name"] = df_ts_efcs[
                    "commerce_name"
                ].str.capitalize()
                df_ts_efcs = (
                    df_ts_efcs.drop(columns="texto")
                    .rename(columns={"commerce_name": "texto"})[
                        [
                            "userid",
                            "grupotipo",
                            "category",
                            "texto",
                            "cantidad",
                            "estabilidad",
                            "recurrencia",
                            "cercania",
                            "tipo",
                            "transacciones",
                        ]
                    ]
                    .groupby("userid")
                    .apply(lambda df: df.to_dict("records"))
                    .reset_index(name=f"ts_indebtness")
                )
                df_efcs = df_efcs.merge(df_ts_efcs, on="userid")
                return df_efcs, df_loans
            except:
                return pd.DataFrame(), df_loans

        def get_bank_loans(df_loans, loan_type):
            def get_loan_dict(row):
                d_loan = {
                    "endBankId": row["bank_name"].capitalize(),
                    "endBankCuot": np.abs(row["deduction"]),
                    "endBankInit": row["initial_balance"],
                    "endBankPend": row["balance"],
                    "endBankTae": row["value_rate"],
                    "endBankAccount": row["account_id"],
                    "endBankBeginDate": str(row["begin_date"]),
                    "endBankEndDate": str(row["end_date"]),
                    "endBankWebAlias": row["webAlias"],
                }
                return d_loan

            df_loans_copy = df_loans.loc[df_loans.loan_type == loan_type]
            s_name = "bankloans" if loan_type == "PRESTAMO" else "mortgage"
            df_loans_copy[f"{s_name}_list"] = df_loans_copy.apply(
                lambda x: get_loan_dict(x), axis=1
            )
            df_loans_totales = (
                df_loans_copy.groupby("userid")
                .agg({"deduction": "sum"})
                .reset_index()
                .rename(columns={"deduction": f"{s_name}_total"})
            )
            df_loans_copy = (
                df_loans_copy.groupby("userid")[f"{s_name}_list"]
                .apply(list)
                .reset_index()
            )
            df_loans_copy = df_loans_copy.merge(
                df_loans_totales, how="left", on="userid"
            )
            return df_loans_copy

        def get_cards(df_cards):
            def get_cards_dict(row):
                d_cards = {
                    "endTarjId": row["card_type"],
                    "endTarjLim": row["limit"],
                    "endTarjDisp": row["disposed"],
                }
                return d_cards

            df_cards["cards_list"] = df_cards.apply(lambda x: get_cards_dict(x), axis=1)
            df_cards_totales = (
                df_cards.groupby("userid")
                .agg({"limit": "sum", "disposed": "sum"})
                .reset_index()
                .rename(columns={"limit": "total_limit", "disposed": "total_disposed"})
            )
            df_cards = (
                df_cards.groupby("userid")["cards_list"].apply(list).reset_index()
            )
            df_cards = df_cards.merge(df_cards_totales, how="left", on="userid")
            return df_cards

        df_original = (
            pd.DataFrame({"userid": df_transactions.userid.to_list()})
            .drop_duplicates()
            .reset_index(drop=True)
        )
        try:
            df_efcs, df_loans = get_efc_loans(df_transactions, df_loans)
            df_original = df_original.merge(
                df_efcs, how="left", on="userid"
            ).reset_index(drop=True)
        except:
            pass

        try:
            df_original = df_original.merge(
                get_bank_loans(df_loans, "PRESTAMO"), how="left", on="userid"
            ).reset_index(drop=True)
        except:
            pass

        try:
            df_original = df_original.merge(
                get_bank_loans(df_loans, "HIPOTECA"), how="left", on="userid"
            ).reset_index(drop=True)
        except:
            pass

        try:
            df_original = df_original.merge(
                get_cards(df_cards), how="left", on="userid"
            ).reset_index(drop=True)
        except:
            pass
        return df_original

    def get_balances(self, df_transactions, df_accounts):
        def get_balances_info(df_transactions, df_accounts, max_days_difference):
            MAX_DATE = df_accounts.valuedate.max()
            MIN_DATE = (
                (
                    dt.strptime(self.s_reference_date, "%Y-%m-%d")
                    - timedelta(days=max_days_difference)
                )
                .date()
                .strftime("%Y-%m-%d")
            )

            def build_aux_df(userid, system_bankid, account_id):
                df_aux = pd.DataFrame()
                df_aux["valuedate"] = (
                    pd.date_range(MIN_DATE, MAX_DATE)
                    .sort_values(ascending=False)
                    .astype(str)
                )
                df_aux["userid"] = userid
                df_aux["system_bankid"] = system_bankid
                df_aux["account_id"] = account_id
                return df_aux.reset_index(drop=True)

            try:
                df_daily_quantities = (
                    df_transactions.query(
                        f"(account_id != '') & (days_difference <= {max_days_difference})"
                    )
                    .groupby(["userid", "system_bankid", "account_id", "valuedate"])
                    .agg({"quantity": "sum"})
                    .sort_values(
                        ["userid", "system_bankid", "account_id", "valuedate"],
                        ascending=False,
                    )
                    .reset_index()
                )
                df_first_transactions = (
                    df_daily_quantities.groupby(
                        ["userid", "system_bankid", "account_id"]
                    )
                    .last()
                    .reset_index()
                )
                df_master_dates = pd.concat(
                    [
                        df
                        for df in df_first_transactions.apply(
                            lambda r: build_aux_df(
                                r.userid, r.system_bankid, r.account_id
                            ),
                            axis=1,
                        )
                    ]
                )
                df_balances = df_master_dates.merge(
                    df_daily_quantities,
                    on=["userid", "system_bankid", "account_id", "valuedate"],
                    how="left",
                ).merge(
                    (
                        df_accounts[
                            ["userid", "account_id", "valuedate", "balance"]
                        ].drop_duplicates(["userid", "account_id"])
                    ),
                    on=["userid", "account_id", "valuedate"],
                    how="left",
                )
                df_balances["quantity"] = df_balances.groupby(
                    ["userid", "system_bankid", "account_id"]
                )["quantity"].shift()
                df_balances["balance"].fillna(-df_balances["quantity"], inplace=True)
                df_balances["balance"] = df_balances.groupby(
                    ["userid", "system_bankid", "account_id"]
                ).agg({"balance": "cumsum"})
                df_balances["balance"] = df_balances.groupby(
                    ["userid", "system_bankid", "account_id"]
                ).agg({"balance": "ffill"})
                df_balances["balance"] = df_balances.groupby(
                    ["userid", "system_bankid", "account_id"]
                ).agg({"balance": "bfill"})
                df_balances = df_balances.query(
                    f'valuedate <= "{self.s_reference_date}"'
                )
                df_account_mean_balances = (
                    df_balances.groupby(["userid", "system_bankid", "account_id"])
                    .agg({"balance": "mean"})
                    .reset_index()
                )
                df_account_last_balances = (
                    df_balances.groupby(["userid", "system_bankid", "account_id"])
                    .first()
                    .reset_index()
                    .rename(columns={"balance": "last_balance"})[
                        ["userid", "system_bankid", "account_id", "last_balance"]
                    ]
                )
                return df_account_mean_balances, df_account_last_balances, df_balances
            except:
                df_exception_1 = pd.DataFrame(
                    data={
                        "userid": df_transactions.userid.unique(),
                        "system_bankid": "0000",
                        "account_id": "0000",
                        "balance": 0.0,
                    }
                )
                df_exception_2 = pd.DataFrame(
                    data={
                        "userid": df_transactions.userid.unique(),
                        "system_bankid": "0000",
                        "account_id": "0000",
                        "valuedate": self.s_reference_date,
                        "balance": 0.0,
                    }
                )
                return df_exception_1, df_exception_1, df_exception_2

        def get_balance_dict(row):
            d_balance = {
                "ACT_ID": f"{row['system_bankid']}_{row['account_id']}",
                "ACT_SALDO": row["balance"],
            }
            return d_balance

        # Balance info in the last 3 months
        (
            df_account_mean_balances_3m,
            df_account_last_balances,
            df_balances,
        ) = get_balances_info(df_transactions, df_accounts, 93)
        # Daily balances
        df_balances["key"] = (
            df_balances["system_bankid"] + "_" + df_balances["account_id"]
        )
        df_balances = (
            df_balances.pivot_table(
                columns=["key"], index=["valuedate"], values=["balance"]
            )
            .ffill(axis=0)
            .bfill(axis=0)
            .round(decimals=2)
        )
        df_balances = df_balances.round(decimals=2)
        df_balances.columns = df_balances.columns.droplevel(0)
        lista = list()
        lista.append(df_balances.to_dict("dict"))
        df_balances = pd.DataFrame(
            {"userid": df_transactions.userid.iloc[0], "balance_transactions": lista}
        )
        # Total last balance
        df_last_balance = (
            df_account_last_balances.groupby("userid")
            .agg({"last_balance": "sum"})
            .reset_index()
        )
        # Total mean balance in the last 3 months
        df_mean_balance_3m = (
            df_account_mean_balances_3m.rename(columns={"balance": "balance_3m"})
            .groupby("userid")
            .agg({"balance_3m": "sum"})
            .reset_index()
        )
        # Mean account balance in the last 3 months
        df_account_mean_balances_3m["balance"] = df_account_mean_balances_3m[
            "balance"
        ].round(2)
        df_account_mean_balances_3m[
            "balances_list"
        ] = df_account_mean_balances_3m.apply(get_balance_dict, axis=1)
        df_account_mean_balances_3m = (
            df_account_mean_balances_3m.groupby(["userid"])
            .agg({"balances_list": list})
            .reset_index()
        )
        # Balance info in the last month
        df_account_mean_balances_1m, _, _ = get_balances_info(
            df_transactions, df_accounts, 31
        )
        # Total mean balance in the last month
        df_mean_balance_1m = (
            df_account_mean_balances_1m.rename(columns={"balance": "balance_1m"})
            .groupby("userid")
            .agg({"balance_1m": "sum"})
            .reset_index()
        )
        # Output
        df_balances = reduce(
            lambda left, rigth: pd.merge(left, rigth, on=["userid"], how="left"),
            [
                df_account_mean_balances_3m,
                df_mean_balance_3m,
                df_mean_balance_1m,
                df_last_balance,
                df_balances,
            ],
        )
        return df_balances

    def get_flags(self, df_transactions, df_incomes, df_balances):
        def get_overdrafts(df_balances, i_threshold_days=5, f_threshold_quantity=-20):
            def get_overdraft_insurance_txs(df, quantity=-3.99):
                """
                Retorna dataframe para casuitica de producto siempre cubierto
                """
                r_overdraft_insurance = r"(siempre cubierto|comisi.n cubierto)"
                df_overdraft_insurance = df.loc[
                    (df["texto_tx"].str.contains(r_overdraft_insurance))
                    & (df["quantity"] == quantity)
                ]
                return df_overdraft_insurance

            def overdraft_insurance(df, max_days=35):
                try:
                    df_overdraft_insurance = get_overdraft_insurance_txs(df)
                    days_since = df_overdraft_insurance.days_difference.values
                    return np.min(days_since) < max_days
                except:
                    return False

            try:
                df_overdraft = (
                    pd.DataFrame(df_balances.balance_transactions[0])
                    .applymap(lambda v: "1" if v < f_threshold_quantity else "0")
                    .apply(lambda c: "1" * i_threshold_days in "".join(c))
                    .reset_index(name="OVERDRAFT_FLAG")
                )
                df_overdraft["userid"] = df_transactions.userid.iloc[0]
                df_overdraft["system_bankid"] = df_overdraft["index"].apply(
                    lambda l: l.split("_")[0]
                )
                df_overdraft["account_id"] = df_overdraft["index"].apply(
                    lambda l: l.split("_")[1]
                )
                df_overdraft.drop(columns="index")
                df_overdraft_insurance = (
                    df_transactions.groupby(
                        by=["userid", "system_bankid", "account_id"]
                    )
                    .apply(overdraft_insurance)
                    .reset_index(name="OVERDRAFT_INSURANCE_FLAG")
                )
                df_overdraft = (
                    df_overdraft.merge(
                        df_overdraft_insurance,
                        on=["userid", "system_bankid", "account_id"],
                    )
                    .groupby(["userid"])
                    .agg({"OVERDRAFT_FLAG": "any", "OVERDRAFT_INSURANCE_FLAG": "any"})
                    .reset_index()
                )
                # Si encontramos producto siempre cubierto entonces generamos listado de transacciones asociadas
                # Si no, genera un listado vacio
                df_tx_overdraft_insurance = get_overdraft_insurance_txs(df_transactions)
                if len(df_tx_overdraft_insurance):
                    df_tx_overdraft_insurance = (
                        df_tx_overdraft_insurance[
                            [
                                "id",
                                "userid",
                                "grupotipo",
                                "category",
                                "system_bankid",
                                "account_id",
                                "card_number",
                                "valuedate",
                                "texto_tx_preclean",
                                "quantity",
                                "relative_balance",
                            ]
                        ]
                        .rename(columns={"texto_tx_preclean": "texto"})
                        .groupby("userid")
                        .apply(lambda df: df.to_dict("records"))
                        .reset_index(name="OVERDRAFT_INSURANCE_TRANSACTIONS")
                    )
                    df_overdraft = df_overdraft.merge(
                        df_tx_overdraft_insurance, on="userid", how="left"
                    ).fillna(0)
                else:
                    df_overdraft["OVERDRAFT_INSURANCE_TRANSACTIONS"] = [[]]
                df_overdraft["OVERDRAFT_DAYS"] = i_threshold_days
                df_overdraft["OVERDRAFT_MONTHS"] = MONTHS_OF_TRANSACTIONS
            except:
                df_overdraft = pd.DataFrame(
                    data={"userid": df_transactions.userid.unique()}
                )
                df_overdraft["OVERDRAFT_DAYS"] = i_threshold_days
                df_overdraft["OVERDRAFT_MONTHS"] = MONTHS_OF_TRANSACTIONS
                return df_overdraft
            return df_overdraft

        def get_bets(df_incomes, max_bets_ratio=0.15):
            def get_bets_ratio(bets, incomes):
                if incomes > 0:
                    return bets / incomes
                else:
                    return 0.0

            try:
                df_tx_bets = df_transactions.loc[
                    (df_transactions.month_difference < 3)
                    & (df_transactions.quantity < 0)
                    & (df_transactions["category"] == "G0508")
                ]
                df_self_employed = is_self_employed()
                if "SELF_EMPLOYED_FLAG" not in df_self_employed.columns:
                    df_self_employed.loc[:, "SELF_EMPLOYED_FLAG"] = False
                if (
                    self.d_offer_info["situacion_laboral"] == "Autónomo"
                ) or df_self_employed["SELF_EMPLOYED_FLAG"].bool():
                    df_tx_bets = df_tx_bets.loc[
                        ~(
                            (df_tx_bets["quantity"] > 3000)
                            & (df_tx_bets["commerce_id"] == 2881)
                        )
                    ]
                df_tx_bets["quantity"] = df_tx_bets["quantity"].abs()
                df_bets = (
                    df_tx_bets.groupby(by=["userid", "month_difference"])
                    .agg({"quantity": "sum"})
                    .rename(columns={"quantity": "bets"})
                    .groupby(by=["userid"])
                    .agg({"bets": "max"})
                    .reset_index()
                )
                df_tx_bets = (
                    df_tx_bets[
                        [
                            "id",
                            "userid",
                            "grupotipo",
                            "category",
                            "system_bankid",
                            "account_id",
                            "card_number",
                            "valuedate",
                            "texto_tx_preclean",
                            "quantity",
                            "relative_balance",
                        ]
                    ]
                    .rename(columns={"texto_tx_preclean": "texto"})
                    .groupby("userid")
                    .apply(lambda df: df.to_dict("records"))
                    .reset_index(name="BETS_TRANSACTIONS")
                )
                df_incomes["incomes"] = df_incomes[
                    ["payroll", "pension", "unemployment"]
                ].sum(axis=1)
                df_bets = (
                    df_bets.merge(
                        df_incomes[["userid", "incomes"]], on="userid", how="left"
                    )
                    .merge(df_tx_bets, on="userid", how="left")
                    .fillna(0)
                )
                df_bets["BETS_RATIO"] = df_bets.apply(
                    lambda r: get_bets_ratio(r.bets, r.incomes), axis=1
                )
                df_bets["BETS_FLAG"] = df_bets["BETS_RATIO"] >= max_bets_ratio
                return df_bets[
                    ["userid", "BETS_FLAG", "BETS_RATIO", "BETS_TRANSACTIONS"]
                ]
            except:
                return pd.DataFrame(data={"userid": df_transactions.userid.unique()})

        def get_seizures():
            r_seizure = re.compile(".*ejec.*emb.*|embargo")
            r_institutions = re.compile(
                ".*" + ".*|.*".join(instituciones_embargo.palabra.to_list()) + ".*"
            )
            r_loan_default = re.compile(".*impag.*prestamo.*")
            r_others = ".*cobro pendiente.*|.*atrasos ptmo.*|.*precio impag.* tarj.*|.*recobro deuda vencida.*"

            def seizure_classification(row):
                if "embargos a lo bestia" in row["texto_tx"]:
                    return None
                if all(
                    [
                        re.match(r_seizure, row["texto_tx"]),
                        re.match(r_institutions, row["texto_tx"]),
                        row["quantity"] > -500,
                        row["quantity"] < -20,
                        row["relative_balance"] <= 0,
                    ]
                ):
                    return "MINOR_SEIZURE"
                if all(
                    [
                        re.match(r_seizure, row["texto_tx"]),
                        not re.match(r_institutions, row["texto_tx"]),
                        row["quantity"] > -150,
                        row["quantity"] < -20,
                        row["relative_balance"] <= 0,
                    ]
                ):
                    return "MINOR_SEIZURE"
                if all(
                    [
                        re.match(r_seizure, row["texto_tx"]),
                        not re.match(r_institutions, row["texto_tx"]),
                        row["quantity"] < -150,
                        row["relative_balance"] <= 0,
                    ]
                ):
                    return "SEIZURE"
                if all(
                    [
                        "recobr" in row["texto_tx"],
                        row["quantity"] < -20,
                        row["relative_balance"] <= 0,
                    ]
                ):
                    return "SEIZURE"
                if all(
                    [
                        re.match(r_loan_default, row["texto_tx"]),
                        row["quantity"] < -20,
                        row["relative_balance"] <= 0,
                    ]
                ):
                    return "SEIZURE"
                if all(
                    [
                        re.match(r_others, row["texto_tx"]),
                        row["quantity"] < -20,
                        row["relative_balance"] <= 0,
                    ]
                ):
                    return "SEIZURE"

            try:
                df_tx_all_seizures = df_transactions.loc[
                    (df_transactions.month_difference < 3)
                    & (df_transactions.quantity < 0)
                    & (df_transactions.grupotipo == "OTROS_EMBARGO")
                ]

                df_tx_all_seizures_all = df_tx_all_seizures.copy()
                df_tx_all_seizures_all = (
                    df_tx_all_seizures_all[
                        [
                            "id",
                            "userid",
                            "grupotipo",
                            "category",
                            "system_bankid",
                            "account_id",
                            "card_number",
                            "valuedate",
                            "texto_tx_preclean",
                            "quantity",
                            "relative_balance",
                        ]
                    ]
                    .rename(columns={"texto_tx_preclean": "texto"})
                    .groupby("userid")
                    .apply(lambda df: df.to_dict("records"))
                    .reset_index(name="SEIZURES_TRANSACTIONS_ALL")
                )

                df_tx_all_seizures["seizure_type"] = df_tx_all_seizures.apply(
                    lambda x: seizure_classification(x), axis=1
                )
                df_tx_all_seizures["SEIZURES_FLAG"] = (
                    df_tx_all_seizures["seizure_type"] == "SEIZURE"
                )
                df_tx_all_seizures["MINOR_SEIZURES_FLAG"] = (
                    df_tx_all_seizures["seizure_type"] == "MINOR_SEIZURE"
                )
                df_seizures = (
                    df_tx_all_seizures.groupby(by=["userid"])
                    .agg({"SEIZURES_FLAG": "any", "MINOR_SEIZURES_FLAG": "any"})
                    .reset_index()
                )

                try:
                    df_tx_seizures = df_tx_all_seizures.query("SEIZURES_FLAG == True")[
                        [
                            "id",
                            "userid",
                            "grupotipo",
                            "category",
                            "system_bankid",
                            "account_id",
                            "card_number",
                            "valuedate",
                            "texto_tx_preclean",
                            "quantity",
                            "relative_balance",
                        ]
                    ].rename(columns={"texto_tx_preclean": "texto"})
                    df_tx_seizures_ = (
                        df_tx_seizures.groupby("userid")
                        .agg({"category": "count", "quantity": "sum"})
                        .rename(
                            columns={
                                "category": "SEIZURES_COUNT",
                                "quantity": "SEIZURES_QUANTITY",
                            }
                        )
                        .reset_index()
                    )
                    df_tx_seizures = (
                        df_tx_seizures.groupby("userid")
                        .apply(lambda df: df.to_dict("records"))
                        .reset_index(name="SEIZURES_TRANSACTIONS")
                    )
                    df_tx_seizures = df_tx_seizures.merge(df_tx_seizures_, on="userid")
                    df_seizures = (
                        df_seizures.merge(df_tx_seizures, on="userid", how="left")
                    ).fillna(0)
                except:
                    df_seizures["SEIZURES_COUNT"] = 0
                    df_seizures["SEIZURES_QUANTITY"] = 0
                    df_seizures["SEIZURES_TRANSACTIONS"] = [[]]

                try:
                    df_tx_seizures = df_tx_all_seizures.query(
                        "MINOR_SEIZURES_FLAG == True"
                    )[
                        [
                            "id",
                            "userid",
                            "grupotipo",
                            "category",
                            "system_bankid",
                            "account_id",
                            "card_number",
                            "valuedate",
                            "texto_tx_preclean",
                            "quantity",
                            "relative_balance",
                        ]
                    ].rename(
                        columns={"texto_tx_preclean": "texto"}
                    )
                    df_tx_seizures_ = (
                        df_tx_seizures.groupby("userid")
                        .agg({"category": "count", "quantity": "sum"})
                        .rename(
                            columns={
                                "category": "MINOR_SEIZURES_COUNT",
                                "quantity": "MINOR_SEIZURES_QUANTITY",
                            }
                        )
                        .reset_index()
                    )
                    df_tx_seizures = (
                        df_tx_seizures.groupby("userid")
                        .apply(lambda df: df.to_dict("records"))
                        .reset_index(name="MINOR_SEIZURES_TRANSACTIONS")
                    )
                    df_tx_seizures = df_tx_seizures.merge(df_tx_seizures_, on="userid")
                    df_seizures = (
                        df_seizures.merge(df_tx_seizures, on="userid", how="left")
                    ).fillna(0)
                except:
                    df_seizures["MINOR_SEIZURES_COUNT"] = 0
                    df_seizures["MINOR_SEIZURES_QUANTITY"] = 0
                    df_seizures["MINOR_SEIZURES_TRANSACTIONS"] = [[]]

                try:
                    df_seizures = (
                        df_seizures.merge(
                            df_tx_all_seizures_all, on="userid", how="left"
                        )
                    ).fillna(0)
                except:
                    df_seizures["SEIZURES_TRANSACTIONS_ALL"] = [[]]

                return df_seizures
            except:
                return pd.DataFrame(data={"userid": df_transactions.userid.unique()})

        def get_returns():
            try:
                # Include here all categories that could contain a returned receipt
                financial_categories = ["G0701", "G0702"]
                non_financial_categories = [
                    "G0101",
                    "G0102",
                    "G0103",
                    "G0104",
                    "G0105",
                    "G0106",
                    "G0107",
                    "G0108",
                    "G0301",
                    "G0302",
                    "G0303",
                    "G0304",
                    "G0305",
                    "G0306",
                    "G0307",
                    "G0398",
                    "G0401",
                    "G0402",
                    "G0403",
                    "G0404",
                    "G0405",
                    "G0406",
                    "G0407",
                    "G0408",
                    "N0198",
                ]
                df_tx_returns = df_transactions.loc[
                    (df_transactions.month_difference < 3)
                    & (
                        df_transactions["grupotipo"]
                        == "DOMICILIACIONES_RECIBOS_DEVUELTOS"
                    )
                    & (
                        df_transactions["category"].isin(
                            financial_categories + non_financial_categories
                        )
                    )
                ]
                df_tx_returns["balance_predevol"] = (
                    df_tx_returns["relative_balance"] - df_tx_returns["quantity"]
                )

                df_tx_returns_all = df_tx_returns.copy()

                df_tx_returns = df_tx_returns.loc[
                    (df_tx_returns.category.isin(financial_categories))
                    | (
                        (df_tx_returns.balance_predevol < 50)
                        & (df_tx_returns.category.isin(non_financial_categories))
                    )
                ]
                df_returns = (
                    df_tx_returns.groupby(by=["userid"])
                    .agg({"id": "count"})
                    .reset_index()
                    .rename(columns={"id": "RETURNS_COUNT"})
                )
                df_returns["RETURNS_FLAG"] = df_returns["RETURNS_COUNT"] > 0
                df_returns["RETURNS_FINANCIAL_COUNT"] = len(
                    df_tx_returns.loc[df_tx_returns.category.isin(financial_categories)]
                )
                df_tx_returns = (
                    df_tx_returns[
                        [
                            "id",
                            "userid",
                            "grupotipo",
                            "category",
                            "system_bankid",
                            "account_id",
                            "card_number",
                            "valuedate",
                            "texto_tx_preclean",
                            "quantity",
                            "relative_balance",
                        ]
                    ]
                    .rename(columns={"texto_tx_preclean": "texto"})
                    .groupby("userid")
                    .apply(lambda df: df.to_dict("records"))
                    .reset_index(name="RETURNS_TRANSACTIONS")
                )

                df_tx_returns_all = (
                    df_tx_returns_all[
                        [
                            "id",
                            "userid",
                            "grupotipo",
                            "category",
                            "system_bankid",
                            "account_id",
                            "card_number",
                            "valuedate",
                            "texto_tx_preclean",
                            "quantity",
                            "relative_balance",
                        ]
                    ]
                    .rename(columns={"texto_tx_preclean": "texto"})
                    .groupby("userid")
                    .apply(lambda df: df.to_dict("records"))
                    .reset_index(name="RETURNS_TRANSACTIONS_ALL")
                )

                df_returns = df_returns.merge(
                    df_tx_returns, on="userid", how="left"
                ).fillna(0)

                try:
                    df_returns = df_returns.merge(
                        df_tx_returns_all, on="userid", how="left"
                    ).fillna(0)
                except:
                    df_returns["RETURNS_TRANSACTIONS_ALL"] = [[]]

                return df_returns
            except:
                return pd.DataFrame(data={"userid": df_transactions.userid.unique()})

        def get_debt_repairs():
            try:
                df_tx_rtd = df_transactions.loc[
                    (df_transactions.commerce_sector == "Deuda")
                    & (df_transactions.quantity < 0)
                ]
                df_rtd = (
                    df_tx_rtd.groupby(by=["userid"])
                    .agg({"id": "count"})
                    .reset_index()
                    .rename(columns={"id": "DEBT_REPAIR_COUNT"})
                )
                df_rtd["DEBT_REPAIR_FLAG"] = df_rtd["DEBT_REPAIR_COUNT"] > 0
                df_tx_rtd = (
                    df_tx_rtd[
                        [
                            "id",
                            "userid",
                            "grupotipo",
                            "category",
                            "system_bankid",
                            "account_id",
                            "card_number",
                            "valuedate",
                            "texto_tx_preclean",
                            "quantity",
                            "relative_balance",
                        ]
                    ]
                    .rename(columns={"texto_tx_preclean": "texto"})
                    .groupby("userid")
                    .apply(lambda df: df.to_dict("records"))
                    .reset_index(name="DEBT_REPAIR_TRANSACTIONS")
                )
                df_rtd = df_rtd.merge(df_tx_rtd, on="userid", how="left").fillna(0)
                df_rtd["DEBT_REPAIR_MONTHS"] = MONTHS_OF_TRANSACTIONS
                return df_rtd
            except:
                df_rtd = pd.DataFrame(data={"userid": df_transactions.userid.unique()})
                df_rtd["DEBT_REPAIR_MONTHS"] = MONTHS_OF_TRANSACTIONS
                return df_rtd

        def is_self_employed():
            try:
                r_self_employed = """
                cuot.+autonomo|r.+e.+autonomo|reautonomo|social.+autonomo|social.+trabajador|s.*s.*autonomo|trabajadores.+autonomo|trabajadores.+aut|tgss.+r.gimen.+general|
                r.gimen.+tgss.+general|r.gimen.+general.+tgss|general.+r.gimen.+tgss|cuotas.+seg|tgss.+cotiza|
                cargo couta ss|rec.aut.+domiciliados navarra|adeudo 303 domiciliado aeat|adeudo 111 domiciliado aeat|cuota tgss regimen general|.+cuotas s.s|
                modelo 130|.+modelo 303|cuotas.+seguridad.+social.+aut|hacienda-modelo 303|social.+autonomo|campo cuota ss periodo liq|
                areadepymes|avalmadrid|cashitapp|claas|descontia|descuentta|descueta facil|famisa|farmafactoring|ficomsa|finanzarel|gedesco|iwoka|
                lico leasing|nobicap|pagaralia|pagares\.net|savia fin|spotcap|targo|urgialis|zencap
                """
                r_self_employed = (
                    "(" + re.sub("(\n)( *)", "", r"%s" % (r_self_employed)) + ")"
                )
                df_tx_self_employed = df_transactions.loc[
                    df_transactions.texto_tx.apply(
                        lambda s: self.search_regex(s, r_self_employed) is not None
                    )
                ]
                df_self_employed = (
                    df_tx_self_employed.groupby(by=["userid"])
                    .agg({"id": "count"})
                    .reset_index()
                    .rename(columns={"id": "SELF_EMPLOYED_COUNT"})
                )
                df_self_employed["SELF_EMPLOYED_FLAG"] = (
                    df_self_employed["SELF_EMPLOYED_COUNT"] > 0
                )
                df_tx_self_employed = (
                    df_tx_self_employed[
                        [
                            "id",
                            "userid",
                            "grupotipo",
                            "category",
                            "system_bankid",
                            "account_id",
                            "card_number",
                            "valuedate",
                            "texto_tx_preclean",
                            "quantity",
                            "relative_balance",
                        ]
                    ]
                    .rename(columns={"texto_tx_preclean": "texto"})
                    .groupby("userid")
                    .apply(lambda df: df.to_dict("records"))
                    .reset_index(name="SELF_EMPLOYED_TRANSACTIONS")
                )
                df_self_employed = df_self_employed.merge(
                    df_tx_self_employed, on="userid", how="left"
                ).fillna(0)
                df_self_employed["SELF_EMPLOYED_MONTHS"] = MONTHS_OF_TRANSACTIONS
                return df_self_employed
            except:
                df_self_employed = pd.DataFrame(
                    data={"userid": df_transactions.userid.unique()}
                )
                df_self_employed["SELF_EMPLOYED_MONTHS"] = MONTHS_OF_TRANSACTIONS
                return df_self_employed

        def in_erte():
            r_erte = "(prest.+sepe)"
            df_erte = df_transactions.copy()
            df_erte["ERTE_FLAG"] = df_erte["texto_tx"].str.contains(r_erte)
            df_erte = df_erte.groupby("userid")["ERTE_FLAG"].any().reset_index()
            return df_erte

        list_dfs = list()
        list_dfs.append(get_overdrafts(df_balances))
        list_dfs.append(get_bets(df_incomes))
        list_dfs.append(get_seizures())
        list_dfs.append(get_returns())
        list_dfs.append(get_debt_repairs())
        list_dfs.append(is_self_employed())
        list_dfs.append(in_erte())
        df_final = reduce(
            lambda left, right: pd.merge(left, right, on="userid", how="left"), list_dfs
        )
        return df_final.fillna(0).drop_duplicates(
            subset=[c for c in df_final.columns if "TRANSACTIONS" not in c]
        )

    def get_microlending(self, df_transactions, i_months_from=2):
        try:
            df_tx_micros = df_transactions.loc[
                (df_transactions.month_difference < i_months_from)
                & (df_transactions.commerce_sector == "Microcreditos")
            ]
            df_micros = (
                df_tx_micros.groupby(by="userid")
                .agg({"commerce_name": "nunique"})
                .reset_index()
                .rename(columns={"commerce_name": "MICRO_LENDING"})
            )
            df_tx_micros_ = (
                df_tx_micros.groupby("userid")
                .agg({"id": "count"})
                .rename(columns={"id": "MICRO_LENDING_COUNT"})
                .reset_index()
            )
            df_tx_micros = (
                df_tx_micros[
                    [
                        "id",
                        "userid",
                        "grupotipo",
                        "category",
                        "system_bankid",
                        "account_id",
                        "card_number",
                        "valuedate",
                        "texto_tx_preclean",
                        "quantity",
                        "relative_balance",
                    ]
                ]
                .rename(columns={"texto_tx_preclean": "texto"})
                .groupby("userid")
                .apply(lambda df: df.to_dict("records"))
                .reset_index(name="MICRO_LENDING_TRANSACTIONS")
            )
            df_micros = (
                df_micros.merge(df_tx_micros, on="userid", how="left")
                .merge(df_tx_micros_, on="userid", how="left")
                .fillna(0)
            )
            df_micros["MICRO_LENDING_MONTHS"] = i_months_from
            return df_micros
        except:
            df_micros = pd.DataFrame(data={"userid": df_transactions.userid.unique()})
            df_micros["MICRO_LENDING_MONTHS"] = i_months_from
            return df_micros

    def get_new_loans(self, df_transactions, df_loans, i_months_from=4):
        df_new_loans = pd.DataFrame(data={"userid": df_transactions.userid.unique()})
        df_new_loans["NEW_LOANS_MONTHS"] = i_months_from
        # Bank loans with begin dates
        try:
            df_banks = df_loans[
                (~df_loans["begin_date"].isna())
                & (df_loans["begin_month"] >= 0)
                & (df_loans["begin_month"] < i_months_from)
                & (df_loans["loan_type"] == "PRESTAMO")
            ]
            df_banks = df_banks.groupby("userid").size().reset_index(name="NL_BANK")
            df_new_loans = df_new_loans.merge(
                df_banks, on="userid", how="outer"
            ).fillna(0)

            # Bank loans without begin date
            def infer_new_loans(df):
                v1 = df["debt"]
                v2 = 1 - (df["debt"] / df["initial_balance"])
                if (v1 > 80000) & (v2 < 0.004):
                    return True
                elif (v1 > 50000) & (v2 < 0.01):
                    return True
                elif (v1 <= 50000) & (v2 < 0.04):
                    return True
                else:
                    return False

            df_banks_wt = df_loans[
                (df_loans["begin_date"].isna()) & (df_loans["loan_type"] == "PRESTAMO")
            ]
            if len(df_banks_wt):
                df_new_loans["NL_BANK_INFERRED"] = df_banks_wt.apply(
                    infer_new_loans, axis=1
                ).sum(axis=0)
            else:
                df_new_loans["NL_BANK_INFERRED"] = 0
        except:
            df_new_loans["NL_BANK"] = 0
            df_new_loans["NL_BANK_INFERRED"] = 0

        try:
            # EFC loans
            df_tx_efcs = df_transactions.loc[
                (df_transactions["month_difference"] < i_months_from)
                & (df_transactions["quantity"] >= 50)
                & (df_transactions["grupotipo"] == "PRESTAMOS_INGRESO")
            ].copy()
            df_tx_efcs["account_id"].fillna("", inplace=True)
            df_tx_efcs["card_number"].fillna("", inplace=True)
            df_tx_efcs.fillna(0)

            df_tx_efcs_list = (
                df_tx_efcs[
                    [
                        "id",
                        "userid",
                        "grupotipo",
                        "category",
                        "system_bankid",
                        "account_id",
                        "card_number",
                        "valuedate",
                        "texto_tx_preclean",
                        "quantity",
                        "relative_balance",
                    ]
                ]
                .rename(columns={"texto_tx_preclean": "texto"})
                .groupby("userid")
                .apply(lambda df: df.to_dict("records"))
                .reset_index(name="NEW_LOANS_TRANSACTIONS")
            )
            df_tx_efcs["efc_detected"] = (
                df_tx_efcs["commerce_sector"] == "Entidades Financieras"
            )
            df_tx_efcs = (
                df_tx_efcs.groupby(["userid", "efc_detected"])
                .size()
                .reset_index(name="NL_EFC")
            )
            df_new_loans = df_new_loans.merge(
                df_tx_efcs.query("efc_detected")[["userid", "NL_EFC"]],
                on="userid",
                how="left",
            ).fillna(0)
            if (df_new_loans["NL_BANK"] + df_new_loans["NL_BANK_INFERRED"])[0] == 0:
                df_new_loans = df_new_loans.merge(
                    df_tx_efcs.query("not efc_detected")[["userid", "NL_EFC"]],
                    on="userid",
                    how="left",
                ).fillna(0)
                df_new_loans["NL_EFC"] = (
                    df_new_loans["NL_EFC_x"] + df_new_loans["NL_EFC_y"]
                )
            df_new_loans = df_new_loans.merge(
                df_tx_efcs_list, on="userid", how="left"
            ).fillna(0)
        except:
            df_new_loans["NL_EFC"] = 0
            df_new_loans["NEW_LOANS_TRANSACTIONS"] = [[]]
        df_new_loans["NL"] = (
            df_new_loans["NL_EFC"]
            + df_new_loans["NL_BANK"]
            + df_new_loans["NL_BANK_INFERRED"]
        )
        return df_new_loans

    def get_transactions_count(self, df_transactions):
        return df_transactions.groupby("userid").size().reset_index(name="n_tx")

    def get_bank_list(self, df_transactions):
        df_banks = df_transactions[["userid", "system_bankid"]].drop_duplicates()
        return (
            df_banks.groupby(by=["userid"])["system_bankid"]
            .apply(list)
            .reset_index()
            .rename({"system_bankid": "bankid_list"})
        )

    def get_regular_expenses(self, df_transactions):
        def get_rent_expenses(df_transactions, min_quantity=-200, max_quantity=-3000):
            pattern_alquiler = r"(renta piso|renta calle|alquil|lloguer)"
            pattern_alquiler_out = r"(coche|casa rural|vacacion)"
            df_alquiler = df_transactions.loc[
                (
                    (df_transactions["texto_tx"].str.contains(pattern_alquiler))
                    | (df_transactions["category"] == "G0201")
                )
                & (~df_transactions["texto_tx"].str.contains(pattern_alquiler_out))
                & (df_transactions["quantity"].between(max_quantity, min_quantity))
            ].copy()
            return df_alquiler

        def get_school_expenses(df_transactions):
            pattern_cole = r"(colegio)"
            pattern_cole_out = r"(oficial|ingenieros|registradores|profesional|gestores|medicos|abogados|fisioterapeutas|administradores|farmacéuticos|comunidad|profesional)"
            df_coles = df_transactions.loc[
                (df_transactions["texto_tx"].str.contains(pattern_cole))
                & (~df_transactions["texto_tx"].str.contains(pattern_cole_out))
                & (df_transactions["quantity"] < 0)
            ].copy()
            return df_coles

        df_transactions_ = df_transactions.loc[
            (df_transactions["quantity"] < 0)
            & (df_transactions["days_difference"] < self.i_days)
        ].copy()

        # Expenses in utilities
        d_dfs = {
            "TELCO": df_transactions_.loc[
                df_transactions_.category.isin(["G0401", "G0402"])
            ],
            "LUZ": df_transactions_.loc[df_transactions_.category == "G0403"],
            "GAS": df_transactions_.loc[df_transactions_.category == "G0404"],
            "AGUA": df_transactions_.loc[df_transactions_.category == "G0405"],
            "OCIO_ENT": df_transactions_.loc[
                df_transactions_.category.isin(["G0407", "G0408"])
            ],
            "COLEGIO": get_school_expenses(df_transactions_),
            "ALQUILER": get_rent_expenses(df_transactions_),
        }
        l_df_expenses = [df_transactions_[["userid"]].drop_duplicates()] + [
            df.groupby(by=["userid", "month_difference"])["quantity"]
            .sum()
            .reset_index()
            .groupby(by=["userid"])["quantity"]
            .mean()
            .reset_index(name=k)
            for k, df in d_dfs.items()
        ]
        df_expenses = reduce(
            lambda left, right: pd.merge(left, right, on="userid", how="left"),
            l_df_expenses,
        ).fillna(0)
        df_expenses["GASTOS_TOTAL"] = df_expenses.loc[
            :, df_expenses.columns != "userid"
        ].sum(axis=1)

        # Adding expenses timelines to response object
        df_ts_expenses = self.treeManager.get_expenses_tree(
            df_transactions, i_days=(3 * 31)
        ).get_df()
        df_expenses = df_expenses.merge(df_ts_expenses, on="userid", how="left")
        return df_expenses

    def get_available_tree_list(self, df_first_tx):
        trees = np.arange(6, df_first_tx["oldest_tx_base_tree"][0] + 1, 3).tolist()
        tree_list = (
            (
                pd.DataFrame(
                    {
                        "tree": np.tile(trees, 1),
                        "userid": np.repeat(df_first_tx["userid"][0], len(trees)),
                    }
                )
            )
            .groupby(by=["userid"])["tree"]
            .apply(list)
            .reset_index()
        )
        tree_list = tree_list if len(tree_list) > 0 else pd.DataFrame({"userid": []})
        return tree_list

    def get_all_frozen(
        self, df_transactions, df_accounts, df_loans, df_cards, df_actives, df_first_tx
    ):
        if len(df_transactions) == 0:
            raise RuntimeError("No transactions left!")
        df_original = (
            pd.DataFrame({"userid": df_transactions.userid.to_list()})
            .drop_duplicates()
            .reset_index(drop=True)
        )
        try:
            df_incomes = self.get_incomes(df_transactions=df_transactions)
            df_original = df_original.merge(
                df_incomes, how="left", on="userid"
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with get_incomes")

        try:
            df_original = df_original.merge(
                self.get_indebtedness(
                    df_transactions=df_transactions,
                    df_loans=df_loans,
                    df_cards=df_cards,
                ),
                how="left",
                on="userid",
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with get_indebtednes")

        try:
            df_balances = self.get_balances(
                df_transactions=df_transactions, df_accounts=df_accounts
            )
            df_original = df_original.merge(
                df_balances, how="left", on="userid"
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with get_balance")

        try:
            df_original = df_original.merge(
                df_actives, how="left", on="userid"
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with df_actives")

        try:
            df_original = df_original.merge(
                self.get_flags(
                    df_transactions=df_transactions,
                    df_incomes=df_incomes,
                    df_balances=df_balances,
                ),
                how="left",
                on="userid",
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with get_flags")

        try:
            df_original = df_original.merge(
                df_first_tx, how="left", on="userid"
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with df_first_tx")

        try:
            df_original = df_original.merge(
                self.get_microlending(df_transactions=df_transactions),
                how="left",
                on="userid",
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with get_microcreditos")

        try:
            df_original = df_original.merge(
                self.get_new_loans(df_transactions=df_transactions, df_loans=df_loans),
                how="left",
                on="userid",
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with get_new_loans")

        try:
            df_original = df_original.merge(
                self.get_transactions_count(df_transactions=df_transactions),
                how="left",
                on="userid",
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with get_transactions_count")

        try:
            df_original = df_original.merge(
                self.get_bank_list(df_transactions=df_transactions),
                how="left",
                on="userid",
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with get_bank_list")

        try:
            df_original = df_original.merge(
                self.get_regular_expenses(df_transactions=df_transactions),
                how="left",
                on="userid",
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with get_regular_expenses")

        try:
            df_original = df_original.merge(
                self.get_available_tree_list(df_first_tx), how="left", on="userid"
            ).reset_index(drop=True)
        except:
            logger.exception("df_original join with get_available_tree_list")
        return df_original
