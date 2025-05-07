## 1. Getting Oriented in Excel

1. **Open the file**

   * Double-click “1. Data Cleaning.xlsx” (or go to **File → Open** and browse to it).
2. **Workbook vs. Worksheet**

   * The overall file is the **workbook**.
   * Each tab at the bottom (e.g. “bike\_buyers”, “Working Sheet”) is a **worksheet**.
3. **The Ribbon**

   * Across the top is the **Ribbon**, organized into tabs: **Home**, **Insert**, **Formulas**, **Data**, **Review**, etc.
   * Hover over any button for a tooltip (“ScreenTip”) explaining what it does.

---

## 2. Visualization Tools

### A. Conditional Formatting

Highlight cells that meet criteria (e.g., high earners).

1. Select the **Income** column (e.g. cells D2\:D1000).
2. On the **Home** tab, click **Conditional Formatting → Highlight Cells Rules → Greater Than…**
3. Enter “60000” and pick a formatting style (e.g. light red fill).
4. Click **OK**: any income over ₹60,000 is now highlighted.

### B. PivotTables & PivotCharts

Summarize and visualize counts by Region, Purchase status, etc.

1. Click any single cell inside your data range on “bike\_buyers”.
2. Go to **Insert → PivotTable**. Accept the defaults and click **OK**.
3. In the new sheet’s PivotTable Fields pane:

   * Drag **Region** to **Rows**.
   * Drag **Purchased Bike** to **Columns**.
   * Drag **ID** (or any field that’s always filled) to **Values**, which defaults to **Count of ID**.
4. To turn this into a chart: with the PivotTable selected, go to **Insert → PivotChart**, choose a **Clustered Column** chart, and click **OK**.
5. Use the chart’s **Design** tab to add titles, adjust colors, and show data labels.

### C. Slicers

Interactive filters for PivotTables.

1. With your PivotTable selected, go to **Insert → Slicer**.
2. Check **Gender**, **Home Owner**, or any field you’d like as a filter.
3. Click **OK**. A floating pane appears—click “Male” or “Female” to instantly filter the pivot.

### D. Sparklines

Tiny in-cell charts showing trends (great if you have time-series data).

1. Suppose you had monthly sales in columns F\:Q.
2. Select the blank cell to the right of your first row of data (e.g. R2).
3. On **Insert → Sparklines**, pick **Line**, set Data Range to F2\:Q2, Location Range to R2, then **OK**.
4. Drag the fill-handle down to copy sparklines for every row.

---

## 3. Key Advanced Formulas

> **Tip:** in Excel you build formulas by starting with an equals sign `=` in any cell.

### A. IF (Logic)

Classify buyers by age:

```excel
=IF(C2 < 35, "Young",
   IF(C2 < 60, "Middle Age", "Old"))
```

* Assumes **Age** is in C2.
* Returns “Young” if age<35, “Middle Age” if 35–59, otherwise “Old”.

### B. VLOOKUP / XLOOKUP (Join-style lookups)

Suppose you build a small table listing Age Brackets (e.g. in G2\:H4), you can use:

```excel
=VLOOKUP(C2, $G$2:$H$4, 2, TRUE)
```

* Looks up C2 in the first column of G2\:H4, returns the matching bracket.

> In modern Excel (Office 365) **XLOOKUP** is even more powerful:

```excel
=XLOOKUP(C2, $G$2:$G$4, $H$2:$H$4, "Not Found", 1)
```

### C. SUMIFS / COUNTIFS (Conditional totals)

Count how many bike-buyers in Europe:

```excel
=COUNTIFS($I:$I, "Europe", $K:$K, "Yes")
```

* Column I = Region, Column K = Purchased Bike.

Sum incomes of owners who purchased a bike:

```excel
=SUMIFS($D:$D, $J:$J, "Yes", $K:$K, "Yes")
```

* Column D = Income, Column J = Home Owner.

### D. INDEX & MATCH (Flexible lookup)

An alternative to VLOOKUP, avoids needing a left-most column:

```excel
=INDEX($H:$H, MATCH(C2, $G:$G, 1))
```

* Finds C2 in G and returns the corresponding H.

### E. Dynamic Array Functions (Office 365)

List unique regions automatically:

```excel
=UNIQUE($I:$I)
```

Sort them:

```excel
=SORT(UNIQUE($I:$I))
```

### F. TEXT Functions

Reformat or combine text. E.g., combine Marital Status and Gender:

```excel
=CONCATENATE(B2, " / ", C2)
```

Or with `&`:

```excel
=B2 & " / " & C2
```

---

## 4. Putting It All Together: A Mini‐Dashboard

1. **Clean & Prepare Data**

   * Use **Remove Duplicates** (Data → Remove Duplicates) to ensure unique IDs.
   * Use **Text to Columns** (Data → Text to Columns) if any field is bunched together.

2. **Add Helper Columns**

   * In a new column “Age Bracket”, enter the IF formula above and fill down.

3. **Create PivotTable**

   * Summarize number of purchasers by Region and Age Bracket.

4. **Insert PivotChart**

   * Add a clustered column or stacked bar chart.

5. **Add Slicers**

   * Attach slicers for Gender and Home Owner to make the dashboard interactive.

6. **Apply Conditional Formatting**

   * On your raw table, highlight high incomes or zero children to spot outliers.

7. **Use Dynamic Arrays**

   * On a separate sheet, type `=UNIQUE(Region)` to get a live list of regions.

8. **Document with Titles & Notes**

   * Right-click your chart title to edit.
   * Insert **Text Boxes** (Insert → Text Box) to explain each chart.

---

### Final Tips

* **Save often**: Excel can crash if a large workbook is open.
* **Explore the Ribbon**: nearly every feature is discoverable there.
* **Use the “Tell me what you want to do” box** (the lightbulb icon) if you’re stuck.
* **Practice**: try variations—e.g. swap in **Average of Income** in your PivotTable to see mean earnings by segment.

With these tools—Conditional Formatting, PivotTables/Charts, Slicers, Sparklines, and advanced formulas like IF, XLOOKUP, SUMIFS, and dynamic arrays—you can turn raw rows into rich insights in minutes. Good luck with your assignment!
