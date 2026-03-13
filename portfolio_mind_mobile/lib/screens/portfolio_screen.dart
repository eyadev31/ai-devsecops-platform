import 'package:flutter/material.dart';
import '../models/portfolio.dart';
import '../services/api_service.dart';

class PortfolioScreen extends StatefulWidget {
  const PortfolioScreen({super.key});

  @override
  State<PortfolioScreen> createState() => _PortfolioScreenState();
}

class _PortfolioScreenState extends State<PortfolioScreen> {
  final ApiService _apiService = ApiService();
  late Future<Portfolio> _futurePortfolio;

  @override
  void initState() {
    super.initState();
    _futurePortfolio = _apiService.getPortfolio(1);
  }

  IconData _getIcon(String assetName) {
    switch (assetName.toLowerCase()) {
      case 'actions us':
        return Icons.show_chart;
      case 'etf':
        return Icons.pie_chart;
      case 'crypto':
        return Icons.currency_bitcoin;
      case 'cash':
        return Icons.account_balance;
      case 'or':
        return Icons.star;
      default:
        return Icons.account_balance_wallet;
    }
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Portfolio>(
      future: _futurePortfolio,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }

        if (snapshot.hasError) {
          return Center(child: Text('Erreur: ${snapshot.error}'));
        }

        if (!snapshot.hasData) {
          return const Center(child: Text('Aucune donnée'));
        }

        final portfolio = snapshot.data!;

        return ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              'Mon portefeuille',
              style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            ...portfolio.assets.map(
              (asset) => Card(
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                child: ListTile(
                  leading: Icon(_getIcon(asset.name)),
                  title: Text(asset.name),
                  subtitle: Text('${asset.percentage} %'),
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}
