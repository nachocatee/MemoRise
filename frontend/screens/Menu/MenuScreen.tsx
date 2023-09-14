import { View, Text, StyleSheet, Image } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import { calculateDynamicWidth } from "../../constants/dynamicSize";
import Colors from "../../constants/colors";

const ModifyInfoScreen = () => {
  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      {/* 유저 닉네임 */}
      <View style={styles.container}>
        <Text style={styles.text}>권소정</Text>
        <View style={styles.emailBox}>
          <Image
            source={require("../../assets/image/kakao_sm.png")}
            style={styles.kakao}
          />
          {/* 유저 이메일 */}
          <Text style={styles.email}>flfk33@naver.com</Text>
        </View>
      </View>
    </LinearGradient>
  );
};

export default ModifyInfoScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
  },
  text: {
    fontSize: calculateDynamicWidth(18),
    fontFamily: "Pretendard-Medium",
    color: Colors.text,
  },
  emailBox: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: Colors.hover,
    width: calculateDynamicWidth(155),
    borderRadius: calculateDynamicWidth(15) / 2,
    opacity: 0.5,
  },
  email: {
    fontSize: calculateDynamicWidth(15),
    color: Colors.text,
  },
  kakao: {
    width: calculateDynamicWidth(12),
    height: calculateDynamicWidth(12),
    marginRight: calculateDynamicWidth(5),
  },
});
