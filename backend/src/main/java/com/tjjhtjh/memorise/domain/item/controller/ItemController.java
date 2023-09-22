package com.tjjhtjh.memorise.domain.item.controller;

import com.tjjhtjh.memorise.domain.item.service.ItemService;
import com.tjjhtjh.memorise.domain.item.service.dto.request.RegistItemRequest;
import com.tjjhtjh.memorise.domain.item.service.dto.response.MemoCountResponse;
import com.tjjhtjh.memorise.domain.item.service.dto.response.RegistItemResponse;
import com.tjjhtjh.memorise.global.file.service.AwsS3Service;
import com.tjjhtjh.memorise.global.file.service.dto.CreateFileRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/items")
public class ItemController {

    private final ItemService itemService;
    private final AwsS3Service awsS3Service;

    private static String dirName = "item-image";

    @PostMapping("/upload")
    public ResponseEntity<CreateFileRequest> uploadMultipleFile(@RequestPart MultipartFile file) {
        return ResponseEntity.ok(awsS3Service.uploadMultiFile(file, dirName));
    }

    @PostMapping
    public ResponseEntity<RegistItemResponse> registItem(@RequestBody RegistItemRequest registItemRequest) {
        itemService.registItem(registItemRequest.getItemSeq(), registItemRequest.getItemImage());
        return ResponseEntity.ok(new RegistItemResponse(true));
    }

    @GetMapping("/{itemSeq}")
    public ResponseEntity<Long> getMemoCountOfItem(@PathVariable Long itemSeq) {
        return ResponseEntity.ok(itemService.getMemoCount(itemSeq));
    }
}
