import { useState } from "react";

export default function InputField({ label, value, onChange, placeholder }) {
  const [focused, setFocused] = useState(false);

  return (
    <div className="field">
      <label className={focused ? "focused" : ""}>{label}</label>
      <input
        value={value}
        placeholder={placeholder}
        onChange={e => onChange(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
      />
    </div>
  );
}