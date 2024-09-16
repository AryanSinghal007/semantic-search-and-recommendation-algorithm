import { Navbar } from "@/components/navbar";
import "./globals.css";

export const metadata = {
  title: "Semantic Search and Recommendation Algorithm",
  description: "Semantic Search and Recommendation Algorithm",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        {children}
        <Navbar/>
      </body>
    </html>
  );
}
