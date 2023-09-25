import React from "react";
import {
  View,
  Image,
  TextInput,
  StyleSheet,
  Pressable,
  Dimensions,
} from "react-native";

import Colors from "../constants/colors";
import { calculateDynamicWidth } from "../constants/dynamicSize";

const screenWidth = Dimensions.get("window").width;

type SearchInputProps = {
  onChangeText: (text: string) => void;
  value: string;
  onSubmitEditing: () => void;
};

const SearchInput: React.FC<SearchInputProps> = ({
  onChangeText,
  value,
  onSubmitEditing,
}) => {
  return (
    <View style={styles.inputContainer}>
      <TextInput
        style={styles.input}
        onChangeText={onChangeText}
        value={value}
        onSubmitEditing={onSubmitEditing}
        returnKeyType="search"
      />
      <Pressable onPress={onSubmitEditing}>
        <Image
          source={require("../assets/icons/search.png")}
          style={styles.image}
        />
      </Pressable>
    </View>
  );
};

export default SearchInput;

const styles = StyleSheet.create({
  inputContainer: {
    width: screenWidth - 60,
    height: calculateDynamicWidth(44),
    backgroundColor: "rgba(255, 255, 255, 0.7)",
    borderRadius: calculateDynamicWidth(15),
    paddingHorizontal: calculateDynamicWidth(15),
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  input: {
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(17),
  },
  image: {
    width: calculateDynamicWidth(17),
    height: calculateDynamicWidth(17),
  },
});
