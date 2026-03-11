import 'package:flutter/material.dart';

class PortfolioScreen extends StatelessWidget {
  const PortfolioScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: ListView(
        children: const [
          Text(
            'Mon portefeuille',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 16),
          Card(
            child: ListTile(
              leading: Icon(Icons.show_chart),
              title: Text('Actions US'),
              subtitle: Text('40 %'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.pie_chart),
              title: Text('ETF'),
              subtitle: Text('20 %'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.currency_bitcoin),
              title: Text('Crypto'),
              subtitle: Text('20 %'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.account_balance),
              title: Text('Cash'),
              subtitle: Text('10 %'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.star),
              title: Text('Or'),
              subtitle: Text('10 %'),
            ),
          ),
        ],
      ),
    );
  }
}
