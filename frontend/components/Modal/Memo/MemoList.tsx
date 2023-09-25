// 라이브러리
import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  Image,
  Pressable,
} from "react-native";
import LinearGradient from "react-native-linear-gradient";

// 스타일
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import Colors from "../../../constants/colors";

// 메인페이지 상태관리를 위한 타입 지정
type MemoListProp = {
  onMemoWritePress: () => void;
  onMemoDetailPress: () => void;
};

const MemoList: React.FC<MemoListProp> = ({
  onMemoWritePress,
  onMemoDetailPress,
}) => {
  // 북마크 상태관리 및 함수
  const [memoData, setMemoData] = useState([
    {
      id: "1",
      date: "2023. 09. 22",
      content:
        "Hate to give the satisfaction, asking how youre doing nowHows the castle built off people you pretend to care about?Just what you wantedLook at you, cool guy, you got it I see the parties and the",
      nickname: "권소정",
      openStatus: "전체공개",
      isBookMark: true,
      isphoto: "",
    },
    {
      id: "2",
      date: "2023. 09. 22",
      content: "테스트2",
      nickname: "김준형",
      openStatus: "비공개",
      isBookMark: false,
      isphoto: "",
    },
    {
      id: "3",
      date: "2023. 09. 22",
      content:
        "테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3테스트3",
      nickname: "김준형",
      openStatus: "비공개",
      isBookMark: false,
      isphoto:
        "https://b106-memorise.s3.ap-northeast-2.amazonaws.com/profile-image/dc971e2c-4a4a-4330-9dfe-ea691ad9bcdf.png",
    },
  ]);

  const changeBookMarkHandler = (id: string) => {
    setMemoData((prevData) =>
      prevData.map((item) =>
        item.id === id ? { ...item, isBookMark: !item.isBookMark } : item
      )
    );
  };

  // FlatList 사용을 위한 Type 지정
  type MemoTypeProps = {
    item: {
      id: string;
      date: string;
      content: string;
      nickname: string;
      openStatus: string;
      isBookMark: boolean;
      isphoto: string;
    };
  };

  // FlatList 사용을 위한 MemoList 정리
  const MemoList: React.FC<MemoTypeProps> = ({ item }) => (
    <View style={styles.memoContainer}>
      <Pressable onPress={onMemoDetailPress}>
        <LinearGradient
          colors={["#FFFFFF", "#F5F5F5"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 0, y: 1 }}
          style={{ minHeight: calculateDynamicWidth(104) }}
        >
          <View style={styles.innerContainer}>
            <View style={styles.calenderContainer}>
              <Text style={styles.calendar}>{item.date}</Text>
            </View>
            {item.isphoto !== "" ? (
              <View>
                <Text
                  style={styles.content}
                  numberOfLines={2}
                  ellipsizeMode="tail"
                >
                  {item.content}
                </Text>
                <Image source={{ uri: item.isphoto }} style={styles.photo} />
              </View>
            ) : (
              <Text
                style={styles.content}
                numberOfLines={4}
                ellipsizeMode="tail"
              >
                {item.content}
              </Text>
            )}
            <View style={styles.bottomContainer}>
              <Text style={styles.nickname}>{item.nickname}</Text>
              <Text style={styles.open}>{item.openStatus}</Text>
            </View>
          </View>
        </LinearGradient>
      </Pressable>
      <Pressable
        onPress={() => changeBookMarkHandler(item.id)}
        style={styles.bookmark}
      >
        {item.isBookMark ? (
          <Image
            source={require("../../../assets/icons/bookmarkblue_fill.png")}
            style={styles.bookmarkSize}
          />
        ) : (
          <Image
            source={require("../../../assets/icons/bookmarkblue.png")}
            style={styles.bookmarkSize}
          />
        )}
      </Pressable>
    </View>
  );

  return (
    <>
      <View style={styles.mainContainer}>
        <Pressable style={styles.memoWriteContainer} onPress={onMemoWritePress}>
          <Image
            source={require("../../../assets/icons/memo_write.png")}
            style={styles.memoWrite}
          />
        </Pressable>
        <View style={styles.memoListContainer}>
          <FlatList
            data={memoData}
            renderItem={({ item }) => <MemoList item={item} />}
            keyExtractor={(item) => item.id}
          />
        </View>
      </View>
    </>
  );
};

export default MemoList;

const styles = StyleSheet.create({
  mainContainer: {
    position: "absolute",
    top: "40%",
    left: "50%",
    transform: [
      { translateY: -calculateDynamicWidth(306) / 2 },
      { translateX: -calculateDynamicWidth(306) / 2 },
    ],
  },

  memoWriteContainer: {
    alignItems: "flex-end",
    marginBottom: 3,
  },

  memoWrite: {
    width: calculateDynamicWidth(38),
    height: calculateDynamicWidth(38),
  },

  memoListContainer: {
    height: calculateDynamicWidth(350),
  },

  memoContainer: {
    width: calculateDynamicWidth(306),
    minHeight: calculateDynamicWidth(104),
    maxHeight: calculateDynamicWidth(185),
    borderRadius: calculateDynamicWidth(15),
    marginBottom: calculateDynamicWidth(12),
    overflow: "scroll",
  },

  innerContainer: {
    margin: calculateDynamicWidth(9),
    flex: 1,
    justifyContent: "space-between",
  },

  bookmark: {
    position: "absolute",
    top: 0,
    marginLeft: calculateDynamicWidth(16),
  },

  bookmarkSize: {
    width: calculateDynamicWidth(17),
    height: calculateDynamicWidth(23),
  },

  calenderContainer: {
    alignItems: "flex-end",
  },

  calendar: {
    color: Colors.hover,
    fontFamily: "Pretendard-Regular",
  },

  content: {
    color: Colors.text,
    fontSize: 18,
    fontFamily: "Pretendard-Regular",
    marginVertical: 5,
    marginHorizontal: 10,
  },

  photo: {
    width: calculateDynamicWidth(275),
    height: calculateDynamicWidth(60),
    borderRadius: calculateDynamicWidth(10),
    marginLeft: calculateDynamicWidth(5),
  },

  bottomContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
  },

  nickname: {
    color: Colors.blue500,
    fontFamily: "Pretendard-Medium",
  },

  open: {
    color: "rgba(76, 106, 255, 0.6)",
    fontFamily: "Pretendard-Medium",
  },
});
