def main(
    df_test_path,
    fold_number="final",
    model_output_path="../data/models",
    views_to_test=None
):
    import pandas as pd
    from Test_RF import load_model_and_evaluate
    from dataset import MiRNADataProcessor
    from Scalar_Product import scalar_product
    processor = MiRNADataProcessor(
        raw_data_path="../data/raw",
        processed_data_path="../data/processed"
    )
    #caricamento dati di train
    processed_path = processor.processed_data_path
    df_train=pd.read_csv(f"{processor.processed_data_path}/df_concat_final.csv", index_col=0)
    complexive_df_path = "../data/processedcomplexive_embeddings_default.csv"
    complexive_df = pd.read_csv(complexive_df_path, index_col=0)
    graph_conv_output_path = f"{processor.processed_data_path}/graph_embeddings.csv"
    node_embeddings = pd.read_csv(graph_conv_output_path, index_col=0)

    # === Caricamento dei dati di test ===
    df_test= pd.read_csv(df_test_path, index_col=0)

    prod_test_path='../data/processed/prod_test.csv'

    df_test = df_test[df_test.columns.intersection(df_train.columns)]

    missing_columns = list(set(df_train.columns) - set(df_test.columns))


    # Se ci sono colonne mancanti, aggiungile tutte in una volta con pd.concat
    if missing_columns:
        new_data = pd.DataFrame(columns=missing_columns, index=df_test.index)
        df_test = pd.concat([df_test, new_data], axis=1)

    scalar_product(df_test, node_embeddings, complexive_df,
                   '../data/processed/prod_test.csv')
    prod_test = pd.read_csv(prod_test_path, index_col=0)

    # === Se non specificato, usa le viste predefinite ===
    if views_to_test is None:
        views_to_test = [
            (['expr_test'], "expr"),
            (['expr_test', 'meta_test'], "expr_meta"),
            (['prod_test', 'meta_test'], "prod_meta"),
            (['expr_test', 'prod_test'], "expr_prod"),
            (['prod_test'], "prod"),
            (['expr_test', 'prod_test', 'meta_test'], "expr_prod_meta"),
            (['meta_test'], "meta")
        ]

    # === Test modelli salvati ===
    for selected_views, view_name in views_to_test:
        print(f"\n🧪 Testing model for view: {view_name}")
        try:
            score, report = load_model_and_evaluate(
                df_test=df_test,
                df_prod_test=prod_test,
                selected_views_test=selected_views,
                fold_number=fold_number,
                model_output_path=model_output_path,
                view_name=view_name
            )
            print(f"✅ Accuracy: {score:.4f}")
            print(f"📝 Classification Report (macro avg): {report['macro avg']}")
        except Exception as e:
            print(f"❌ Failed to evaluate model for view {view_name}: {e}")


if __name__ == "__main__":
    custom_views = [
        (['expr_test', 'prod_test'], "expr_prod"),

    ]

    main(
        df_test_path='../data/processed/df_89_tot.csv',
        fold_number="final",
        model_output_path="../data/models",
        views_to_test=custom_views
    )