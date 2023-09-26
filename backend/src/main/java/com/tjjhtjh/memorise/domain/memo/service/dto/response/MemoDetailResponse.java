package com.tjjhtjh.memorise.domain.memo.service.dto.response;

import com.tjjhtjh.memorise.domain.memo.repository.entity.AccessType;
import com.tjjhtjh.memorise.domain.tag.service.dto.response.TaggedUserResponse;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.nio.file.LinkOption;
import java.time.LocalDateTime;
import java.util.List;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MemoDetailResponse {

    private String content;
    private String nickname;
    private LocalDateTime updatedAt;
    private String itemName;
    private AccessType accessType;
    private String file;
    private String itemImage;
    private Boolean isBookmarked;
    private List<TaggedUserResponse> taggedUserList;

    public MemoDetailResponse detailResponse(MemoDetailResponse reponse, Boolean bookmarkCheck, List<TaggedUserResponse> userList) {
        return MemoDetailResponse.builder()
                .content(reponse.getContent())
                .nickname(reponse.getNickname())
                .itemName(reponse.getItemName())
                .updatedAt(reponse.getUpdatedAt())
                .accessType(reponse.getAccessType())
                .file(reponse.getFile())
                .itemImage(reponse.getItemImage())
                .isBookmarked(bookmarkCheck)
                .taggedUserList(userList)
                .build();
    }

}
